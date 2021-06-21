#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h> 
#include <crtdbg.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <random>

#include "..\numpy.hpp"
#include "..\lightPath.hpp"
#include "..\plyio.hpp"
#include "..\optimization.hpp"

#include "ceres\ceres.h"
#include "glog\logging.h"

using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

int resX = 2048, resY = 1536;

enum INITIAL_DEPTH {
	CONSTANT,
	TRIANGULATION,
	DFT,
};

bool useFirstImageAsReference = true;
INITIAL_DEPTH initialDepthAlgorithm = TRIANGULATION;
// Single-viewpointの場合はfalseを強く推奨
bool useDepthRepresentation = false;
bool rs_static = false;
bool cam_static = false;
bool use_softConst = false;
double lambda_soft = 1.0;
double range_soft = 200;
double range_interp = 50;

std::random_device rd;
std::mt19937 mt(rd());

void readParams(const std::string& filename) {
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cout << filename << " not found." << std::endl;
		return;
	}

	resX = (int) fs["ResX"];
	resY = (int) fs["ResY"];

	useFirstImageAsReference = (int) fs["UseFirstImageAsReferenfe"];
	initialDepthAlgorithm = (INITIAL_DEPTH) (int) fs["InitialDepthAlgorithm"];
	useDepthRepresentation = (int) fs["UseDepthRepresentation"];
	rs_static = (int) fs["rs_static"];
	cam_static = (int) fs["cam_static"];
	use_softConst = (int) fs["UseSoftConstraint"];
	lambda_soft = (double) fs["Lambda_SoftConstraint"];
	range_soft = (double) fs["Range_SoftConstraint"];
	range_interp = (double) fs["Range_Interpolation"];

	if (rs_static && cam_static) {
		std::cerr << "Refractive surface and camera cannot be static simultaneously." << std::endl;
		exit(0);
	}
	if (cam_static && useDepthRepresentation) {
		std::cerr << "Depth representation will not work correctly under static camera." << std::endl;
		exit(0);
	}
	if (use_softConst && useDepthRepresentation) {
		std::cerr << "Soft constraint cannot be used in depth representation." << std::endl;
		exit(0);
	}
}

int countNumObservedImages(int index, const std::vector<std::map<int, Eigen::Vector2d>>& point2DList) {
	int count = 0;
	for (const auto& image : point2DList) {
		count += image.find(index) != image.end();
	}
	return count;
}

int main(int argc, char* argv[]) {
	_CrtDumpMemoryLeaks();
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);

	if (argc < 5) {
		std::cout << "Usage: numImages, param.yml, camMat, point2Dformat, texture (optional)" << std::endl;
		return 0;
	}

	int numImages = std::stoi(argv[1]);
	readParams(argv[2]);

	bool useTexture = false;
	cv::Mat tex;
	if (argc >= 6) {
		tex = cv::imread(argv[5], cv::IMREAD_COLOR);
		if (!tex.empty()) {
			useTexture = true;
		}
	}

	// camera setting
	// Prepare camera
	cv::Mat camMat(3, 3, CV_64F), camDist = cv::Mat::zeros(1, 4, CV_64F);
	std::ifstream ifs_camMat(argv[3]);
	if (!ifs_camMat) {
		std::cout << "Camera matrix file " << argv[2] << " not found" << std::endl;
		return 0;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			ifs_camMat >> camMat.at<double>(i, j);
		}
	}

	realCamera cam(camMat, camDist);
	cam.calcCamParams({ resX, resY }, { 0.0071, 0.0054 });

//	camMat.at<float>(0, 0) *= r_w<float>;
//	camMat.at<float>(1, 1) *= r_w<float>;

	// Load data
	std::vector<std::map<int, Eigen::Vector2d>> point2DList;
	std::vector<int> indexList;
	int refImage = 0;
	for (int i = 0; i < numImages; i++) {
		char filename[FILENAME_MAX];
		sprintf_s(filename, argv[4], i);
		std::ifstream ifs_data(filename);

		if (!ifs_data) {
			std::cout << filename << " not found" << std::endl;
			continue;
		}

		std::map<int, Eigen::Vector2d> point2Dmap;

		while (true) {
			int index;
			double x, y;
			ifs_data >> index >> x >> y;
			if (ifs_data.eof()) {
				break;
			}
			point2Dmap.insert(std::make_pair(index, Eigen::Vector2d(x, y)));
			indexList.push_back(index);
		}

		point2DList.emplace_back(point2Dmap);

		// 最初の画像をリファレンスにしない場合は点数が最大の画像をリファレンスとする
		if (!useFirstImageAsReference && point2DList.rbegin()->size() > point2DList.at(refImage).size()) {
			refImage = i;
		}
	}

	std::sort(indexList.begin(), indexList.end());
	indexList.erase(std::unique(indexList.begin(), indexList.end()), indexList.end());

	numImages = point2DList.size();

	//
	// 初期値計算
	//
	std::map<int, Eigen::Vector3d> point3DList_init_ave;

	if (initialDepthAlgorithm == INITIAL_DEPTH::CONSTANT) {
		// 初期値を定数で与える
		for (const auto& i : indexList) {
			if (point2DList.at(refImage).find(i) == point2DList.at(refImage).end()) {
				continue;
			}
			auto ray = cam.castRay(point2DList.at(refImage).at(i));
			point3DList_init_ave.insert(std::make_pair(i, ray.origin + ray.dir * 1.0 / ray.dir.z()));
		}
	} else if (initialDepthAlgorithm == INITIAL_DEPTH::TRIANGULATION) {
		// 5-points algorithmによる初期3次元点計算(ここでのカメラRtには大した意味はない)
		std::vector<std::map<int, Eigen::Vector3d>> point3DList_init(numImages);
		for (int i = 0; i < numImages; i++) {
			if (i == refImage) {
				continue;
			}

			std::vector<cv::Point2d> pts_ref, pts_img;
			std::vector<int> pts_index;
			for (const auto& f : point2DList.at(refImage)) {
				if (point2DList.at(i).find(f.first) == point2DList.at(i).end()) {
					continue;
				}
				const auto& pts = point2DList.at(i).at(f.first);
				pts_ref.emplace_back(f.second.x(), f.second.y());
				pts_img.emplace_back(pts.x(), pts.y());
				pts_index.push_back(f.first);
			}

	/*		cv::Mat debugImg(resY / 2, resX / 2, CV_8UC3);
			for (int i = 0; i < pts_ref.size(); i++) {
				cv::circle(debugImg, pts_ref.at(i) / 2, 3, cv::Scalar(255, 0, 0));
				cv::circle(debugImg, pts_img.at(i) / 2, 3, cv::Scalar(0, 0, 255));
				cv::line(debugImg, pts_ref.at(i) / 2, pts_img.at(i) / 2, cv::Scalar(255, 255, 255));
			}

			cv::imshow("", debugImg);
			cv::waitKey();*/

			cv::Mat EMat = cv::findEssentialMat(pts_ref, pts_img, camMat);

			// recoverPose
			cv::Mat r, t;
			cv::recoverPose(EMat, pts_ref, pts_img, camMat, r, t);

			// triangulation
			cv::Mat pts_homo;
			cv::Mat projMat_ref(3, 4, CV_64F), projMat_img(3, 4, CV_64F);
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					projMat_ref.at<double>(i, j) = i == j ? 1.0 : 0.0;
					projMat_img.at<double>(i, j) = r.at<double>(i, j);
				}
				projMat_ref.at<double>(i, 3) = 0.0;
				projMat_img.at<double>(i, 3) = t.at<double>(i);
			}

			cv::triangulatePoints(camMat * projMat_ref, camMat * projMat_img, pts_ref, pts_img, pts_homo);

			cv::Mat pts_init;
			cv::convertPointsFromHomogeneous(pts_homo.t(), pts_init);
			for (int r = 0; r < pts_init.rows; r++) {
				Eigen::Vector3d pts;
				cv::cv2eigen(pts_init.at<cv::Vec3d>(r), pts);
				point3DList_init.at(i).insert(std::make_pair(pts_index.at(r), pts));
			}
		}

		// 各画像の初期三次元形状の正規化
		for (int i = 0; i < numImages; i++) {
			double averageDepth = 0.0;
			for (const auto& it : point3DList_init.at(i)) {
				averageDepth += it.second.z();
			}
			averageDepth /= point3DList_init.at(i).size();
			for (auto& it : point3DList_init.at(i)) {
				it.second /= averageDepth;
			}
		}

		// 平均形状の算出と使う画像の選定
		std::vector<int> activeImageList;
		for (int i = 0; i < numImages; i++) {
			if (i == refImage) {
				continue;
			}
			activeImageList.push_back(i);
		}

		bool activeImageChanged = false;
		do {
			// 平均形状の計算
			point3DList_init_ave.clear();
			for (const auto& index : indexList) {
				Eigen::Vector3d center(0, 0, 0);
				int count = 0;
				for (const auto& i : activeImageList) {
					if (point3DList_init.at(i).find(index) == point3DList_init.at(i).end()) {
						continue;
					}
					center += point3DList_init.at(i).at(index);
					count++;
				}
				if (count <= 0) {
					continue;
				}
				point3DList_init_ave.insert(std::make_pair(index, center / count));
			}

			// 平均分散と平均分散の平均を求める
			double aveaveNorm = 0.0;
			std::map<int, double> aveNormList;
			for (const auto& i : activeImageList) {
				double aveNorm = 0.0;
				int count = 0;
				for (const auto& index : indexList) {
					if (point3DList_init_ave.find(index) == point3DList_init_ave.end() ||
						point3DList_init.at(i).find(index) == point3DList_init.at(i).end()) {
						continue;
					}
					aveNorm += (point3DList_init.at(i).at(index) - point3DList_init_ave.at(index)).squaredNorm();
					count++;
				}
				aveNorm /= count;
				aveNormList.insert(std::make_pair(i, aveNorm));

				aveaveNorm += aveNorm;
			}
			aveaveNorm /= activeImageList.size();

			// 平均の平均分散より遥かに平均分散が大きい画像を弾く
			activeImageChanged = false;
			auto it = activeImageList.begin();
			while (it != activeImageList.end()) {
				if (aveNormList.at(*it) / aveaveNorm > 2.0) {
					it = activeImageList.erase(it);
					activeImageChanged = true;
				} else {
					it++;
				}
			}
		} while (activeImageChanged);

		std::cout << "Use images: ";
		for (const auto& it : activeImageList) {
			std::cout << it << ", ";
		}
		std::cout << std::endl;
	} else if (initialDepthAlgorithm == INITIAL_DEPTH::DFT) {
		// DfTによる初期値計算
		std::vector<std::pair<int, double>> residualList;
		double maxResidual = 0.0;
		for (const auto& j : indexList) {
			Eigen::Vector2d center(0, 0);
			int count = 0;
			for (int i = 0; i < numImages; i++) {
				if (point2DList.at(i).find(j) == point2DList.at(i).end()) {
					continue;
				}
				center += point2DList.at(i).at(j);
				count++;
			}
			// 全画像で観測されていなかったら信用できないので外す
			if (count < numImages) {
				continue;
			}
			center /= count;
			double residual = 0;
			for (int i = 0; i < numImages; i++) {
				if (point2DList.at(i).find(j) == point2DList.at(i).end()) {
					continue;
				}
				residual += (point2DList.at(i).at(j) - center).squaredNorm();
			}
			residual = residual / count;
			maxResidual = std::max(maxResidual, residual);
			residualList.emplace_back(std::make_pair(j, residual));
		}

		for (const auto& it : residualList) {
			if (point2DList.at(refImage).find(it.first) == point2DList.at(refImage).end()) {
				continue;
			}
			auto ray = cam.castRay(point2DList.at(refImage).at(it.first));
			point3DList_init_ave.insert(std::make_pair(it.first, ray.origin + ray.dir * it.second / maxResidual));
		}
	} else {
		std::cout << "Unknown algorithm specified." << std::endl;
		return 0;
	}

	// PnPによる初期カメラRt計算
	std::vector<Eigen::Vector3d> tvecs_init;
	std::vector<Eigen::Quaterniond> rquats_init;
	for (int i = 0; i < numImages; i++) {
		if (i == refImage) {
			tvecs_init.emplace_back(Eigen::Vector3d(0, 0, 0));
			rquats_init.emplace_back(Eigen::Quaterniond::Identity());
			continue;
		}

		std::vector<cv::Point2f> point2DListCV;
		std::vector<cv::Point3f> point3DListCV;
		for (const auto& it : point2DList.at(i)) {
			if (point3DList_init_ave.find(it.first) == point3DList_init_ave.end()) {
				continue;
			}
			point2DListCV.emplace_back(it.second.x(), it.second.y());
			const auto& pos = point3DList_init_ave.at(it.first);
			point3DListCV.emplace_back(pos.x(), pos.y(), pos.z());
		}
		cv::Mat r, t;
//		cv::solvePnP(point3DListCV, point2DListCV, camMat, camDist, r, t);
		cv::solvePnPRansac(point3DListCV, point2DListCV, camMat, camDist, r, t, false, 100, 8.0);

		cv::Mat rmat_cv;
		cv::Rodrigues(r, rmat_cv);
		Eigen::Matrix3d rmat;
		cv::cv2eigen(rmat_cv, rmat);
		rquats_init.emplace_back(Eigen::Quaterniond(rmat));
		tvecs_init.emplace_back(t.at<double>(0), t.at<double>(1), t.at<double>(2));
	}

	// 基準となるDepthを求める
	Eigen::Vector3d nearestPos(DBL_MAX, DBL_MAX, DBL_MAX);
	for (auto& it : point3DList_init_ave) {
		if (it.second.z() < nearestPos.z()) {
			nearestPos = it.second;
		}
	}

	// 再投影誤差が大きい画像は除く
	std::cout << "Initial RMSE: " << std::endl;
	std::vector<int> activeImageList;
	for (int i = 0; i < numImages; i++) {
		if (i == refImage) {
			activeImageList.push_back(i);
			continue;
		}
//		cv::Mat img(resY, resX, CV_8UC3);
//		for (const auto& it : point2DList.at(i)) {
//			cv::circle(img, cv::Point2d(it.second.x(), it.second.y()), 3, cv::Scalar(255, 0, 0));
//		}
		double mse = 0.0;
		int count = 0;
		for (const auto& it : point3DList_init_ave) {
			if (point2DList.at(i).find(it.first) == point2DList.at(i).end()) {
				continue;
			}
			Eigen::Vector3d pos3D = rquats_init.at(i) * it.second + tvecs_init.at(i);
//			plane<double> pl(Eigen::Vector3d(0.0, 0.0, 1.0), nearestPos.z());
//			auto pos2D = cam.project(forward(pos3D, pl));
			auto pos2D = cam.project(pos3D);
			mse += (pos2D - point2DList.at(i).at(it.first)).squaredNorm();
			count++;
//			cv::circle(img, cv::Point2d(pos2D.x(), pos2D.y()), 3, cv::Scalar(0, 0, 255));
		}
		double rmse = sqrt(mse / count);
		std::cout << "Image " << i << " : " << rmse << std::endl;
		if (rmse > 100) {
			rquats_init.at(i) = Eigen::Quaterniond::Identity();
			tvecs_init.at(i) = Eigen::Vector3d(0.0, 0.0, 0.0);
			continue;
		}
		activeImageList.push_back(i);
//		cv::imwrite("reproj" + std::to_string(i) + ".png", img);
	}

	//
	// 最適化
	//

	// 最適化パラメータ: R(3), t(3), P(3), n(2), d(1)
	// ただしRtはref imageに関しては固定
	double*	cam_r = new double[3 * numImages];
	double* cam_t = new double[3 * numImages];
	double* d_p = new double[indexList.size()];
	double* sf_p = new double[indexList.size() * 3];
	double* rs_n = new double[2 * numImages];
	double* rs_d = new double[numImages];
	double* sf_n = new double[numImages * indexList.size() * 2];

	// 初期値の設定
	for (int i = 0; i < numImages; i++) {
		const auto rmat = rquats_init.at(i).normalized().toRotationMatrix();
		cv::Mat rmat_cv;
		cv::eigen2cv(rmat, rmat_cv);
		cv::Mat rvec;
		cv::Rodrigues(rmat_cv, rvec);

		cam_r[i * 3 + 0] = rvec.at<double>(0);
		cam_r[i * 3 + 1] = rvec.at<double>(1);
		cam_r[i * 3 + 2] = rvec.at<double>(2);

		cam_t[i * 3 + 0] = tvecs_init.at(i).x();
		cam_t[i * 3 + 1] = tvecs_init.at(i).y();
		cam_t[i * 3 + 2] = tvecs_init.at(i).z();

		rs_n[i * 2 + 0] = 0.0;
		rs_n[i * 2 + 1] = 0.0;

		std::fill_n(&sf_n[i * indexList.size() * 2], indexList.size() * 2, 0.0);

		rs_d[i] = nearestPos.z() / 2.0;
	}

	for (int i = 0; i < indexList.size(); i++) {
		if (point3DList_init_ave.find(indexList.at(i)) == point3DList_init_ave.end()) {
			// 近傍の点を使って補間
			int useImage = -1;
			Eigen::Vector2d pos2D;
			for (int j = 0; j < numImages; j++) {
				if (point2DList.at(j).find(indexList.at(i)) != point2DList.at(j).end()) {
					useImage = j;
					pos2D = point2DList.at(j).at(indexList.at(i));
					break;
				}
			}
			Eigen::Vector3d avePos(0, 0, 0);
			int count = 0;
			for (const auto& j : indexList) {
				if (point3DList_init_ave.find(j) == point3DList_init_ave.end() ||
					point2DList.at(useImage).find(j) == point2DList.at(useImage).end() ||
					(pos2D - point2DList.at(useImage).at(j)).norm() > 300) {
					continue;
				}
				avePos += point3DList_init_ave.at(j);
				count++;
			}
			if (useDepthRepresentation) {
				if (count <= 0) {
					d_p[i] = nearestPos.z();
				} else {
					d_p[i] = avePos.z() / count;
				}
			} else {
				if (count <= 0) {
					sf_p[i * 3 + 0] = nearestPos.x();
					sf_p[i * 3 + 1] = nearestPos.y();
					sf_p[i * 3 + 2] = nearestPos.z();
				} else {
					sf_p[i * 3 + 0] = avePos.x() / count;
					sf_p[i * 3 + 1] = avePos.y() / count;
					sf_p[i * 3 + 2] = avePos.z() / count;
				}
			}
		} else {
			if (useDepthRepresentation) {
				d_p[i] = point3DList_init_ave.at(indexList.at(i)).z();
			} else {
				sf_p[i * 3 + 0] = point3DList_init_ave.at(indexList.at(i)).x();
				sf_p[i * 3 + 1] = point3DList_init_ave.at(indexList.at(i)).y();
				sf_p[i * 3 + 2] = point3DList_init_ave.at(indexList.at(i)).z();
			}
		}
	}

	Problem problem;
	for (int i = 0; i < numImages; i++) {
		if (useDepthRepresentation) {
			if (i == refImage) {
				continue;
			}
			for (int j = 0; j < indexList.size(); j++) {
				if (point2DList.at(i).find(indexList.at(j)) == point2DList.at(i).end() ||
					point2DList.at(refImage).find(indexList.at(j)) == point2DList.at(refImage).end()) {
					continue;
				}
				if (rs_static) {
					ceres::CostFunction* cost_function = HardForwardFunctorDepthRsStatic::Create(point2DList.at(i).at(indexList.at(j)), point2DList.at(refImage).at(indexList.at(j)), cam);
					problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &cam_r[i * 3], &cam_t[i * 3], &d_p[j], &rs_n[refImage * 2], &rs_d[refImage]);
				} else if (cam_static) {
					ceres::CostFunction* cost_function = HardForwardFunctorDepthCamStatic::Create(point2DList.at(i).at(indexList.at(j)), point2DList.at(refImage).at(indexList.at(j)), cam);
					problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &d_p[j], &rs_n[refImage * 2], &rs_d[refImage], &rs_n[i * 2], &rs_d[i]);
				} else {
					ceres::CostFunction* cost_function = HardForwardFunctorDepth::Create(point2DList.at(i).at(indexList.at(j)), point2DList.at(refImage).at(indexList.at(j)), cam);
					problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &cam_r[i * 3], &cam_t[i * 3], &d_p[j], &rs_n[refImage * 2], &rs_d[refImage], &rs_n[i * 2], &rs_d[i]);
				}
				problem.SetParameterLowerBound(&d_p[j], 0, 0.01);
			}
		} else {
			for (int j = 0; j < indexList.size(); j++) {
				if (point2DList.at(i).find(indexList.at(j)) == point2DList.at(i).end() ||
					countNumObservedImages(indexList.at(j), point2DList) < 2) {
					continue;
				}
				if (use_softConst) {
					if (rs_static) {
						ceres::CostFunction* cost_function = SoftForwardFunctor::Create(point2DList.at(i).at(indexList.at(j)), cam);
						problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &cam_r[i * 3], &cam_t[i * 3], &sf_p[j * 3], &rs_n[refImage * 2], &rs_d[refImage], &sf_n[(i * indexList.size() + j) * 2]);
					} else if (cam_static) {
						ceres::CostFunction* cost_function = SoftForwardFunctorCamStatic::Create(point2DList.at(i).at(indexList.at(j)), cam);
						problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &sf_p[j * 3], &rs_n[i * 2], &rs_d[i], &sf_n[(i * indexList.size() + j) * 2]);
					} else {
						ceres::CostFunction* cost_function = SoftForwardFunctor::Create(point2DList.at(i).at(indexList.at(j)), cam);
						problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &cam_r[i * 3], &cam_t[i * 3], &sf_p[j * 3], &rs_n[i * 2], &rs_d[i], &sf_n[(i * indexList.size() + j) * 2]);
					}
					if (i == refImage) {
						// リファレンス画像は完全に波が無いと仮定してconstantにする
						problem.SetParameterBlockConstant(&sf_n[(i * indexList.size() + j) * 2]);
					} else {
						for (int k = 0; k < 2; k++) {
							problem.SetParameterLowerBound(&sf_n[(i * indexList.size() + j) * 2], k, -1.0);
							problem.SetParameterUpperBound(&sf_n[(i * indexList.size() + j) * 2], k, 1.0);
						}

						// Regularizer
						for (int k = j + 1; k < indexList.size(); k++) {
							if (point2DList.at(i).find(indexList.at(k)) == point2DList.at(i).end() ||
								countNumObservedImages(indexList.at(k), point2DList) < 2) {
								continue;
							}
							double distance = (point2DList.at(i).at(indexList.at(k)) - point2DList.at(i).at(indexList.at(j))).norm();
							if (distance > range_soft) {
								continue;
							}
							ceres::CostFunction* cost_function = NormalRegularizer::Create(lambda_soft * (range_soft - distance));
							problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &sf_n[(i * indexList.size() + j) * 2], &sf_n[(i * indexList.size() + k) * 2]);
						}
					}
				} else {
					if (rs_static) {
						ceres::CostFunction* cost_function = HardForwardFunctor::Create(point2DList.at(i).at(indexList.at(j)), cam);
						problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &cam_r[i * 3], &cam_t[i * 3], &sf_p[j * 3], &rs_n[refImage * 2], &rs_d[refImage]);
					} else if (cam_static) {
						ceres::CostFunction* cost_function = HardForwardFunctorCamStatic::Create(point2DList.at(i).at(indexList.at(j)), cam);
						problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &sf_p[j * 3], &rs_n[i * 2], &rs_d[i]);
					} else {
						ceres::CostFunction* cost_function = HardForwardFunctor::Create(point2DList.at(i).at(indexList.at(j)), cam);
						problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &cam_r[i * 3], &cam_t[i * 3], &sf_p[j * 3], &rs_n[i * 2], &rs_d[i]);
					}
				}
				problem.SetParameterLowerBound(&sf_p[j * 3], 2, 0.01);
			}
		}
	}

	if (rs_static) {
		for (int j = 0; j < 2; j++) {
			problem.SetParameterLowerBound(&rs_n[refImage * 2], j, -1.0);
			problem.SetParameterUpperBound(&rs_n[refImage * 2], j, 1.0);
		}
		problem.SetParameterLowerBound(&rs_d[refImage], 0, 0.01);
	} else {
		for (int i = 0; i < numImages; i++) {
			for (int j = 0; j < 2; j++) {
				problem.SetParameterLowerBound(&rs_n[i * 2], j, -1.0);
				problem.SetParameterUpperBound(&rs_n[i * 2], j, 1.0);
			}
			problem.SetParameterLowerBound(&rs_d[i], 0, 0.01);
		}
	}
	if (!cam_static && !useDepthRepresentation) {
		problem.SetParameterBlockConstant(&cam_r[refImage * 3]);
		problem.SetParameterBlockConstant(&cam_t[refImage * 3]);
	}
	problem.SetParameterBlockConstant(&rs_d[refImage]);

	Solver::Options option;
	option.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
	option.minimizer_progress_to_stdout = true;
	option.max_num_iterations = 500;
	option.function_tolerance = 1e-16;
	option.gradient_tolerance = 1e-16;
	option.parameter_tolerance = 1e-16;
	Solver::Summary summary;
	Solve(option, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	// データを回収
	std::vector<Eigen::Vector3d> tvecs;
	std::vector<Eigen::Quaterniond> rquats;
	std::vector<std::pair<Eigen::Vector3d, double>> nd;

	for (int i = 0; i < numImages; i++) {
		if (cam_static) {
			rquats.emplace_back(Eigen::Quaterniond::Identity());
			tvecs.emplace_back(Eigen::Vector3d(0, 0, 0));
		} else {
			cv::Mat rvec = (cv::Mat_<double>(3, 1) << cam_r[i * 3 + 0], cam_r[i * 3 + 1], cam_r[i * 3 + 2]);
			cv::Mat rmat_cv;
			cv::Rodrigues(rvec, rmat_cv);
			Eigen::Matrix3d rmat;
			cv::cv2eigen(rmat_cv, rmat);

			rquats.emplace_back(Eigen::Quaterniond(rmat));
			tvecs.emplace_back(cam_t[i * 3 + 0], cam_t[i * 3 + 1], cam_t[i * 3 + 2]);
		}
		if (!rs_static || i == refImage) {
			nd.emplace_back(std::make_pair(Eigen::Vector3d(rs_n[i * 2 + 0], rs_n[i * 2 + 1], 1).normalized(), rs_d[i]));
		}
	}

	std::map<int, Eigen::Vector3d> point3DList;
	for (int i = 0; i < indexList.size(); i++) {
		if (point2DList.at(refImage).find(indexList.at(i)) == point2DList.at(refImage).end() ||
			countNumObservedImages(indexList.at(i), point2DList) < 2) {
			continue;
		}
		if (useDepthRepresentation) {
			auto pos2D = point2DList.at(refImage).at(indexList.at(i));
			plane<double> pl(Eigen::Vector3d(rs_n[refImage * 2 + 0], rs_n[refImage * 2 + 1], 1.0).normalized(), rs_d[refImage]);
			auto ray = backward(pos2D, pl, cam);
			auto pos3D = ray.origin + ray.dir * d_p[i] / ray.dir.z();
			point3DList.insert(std::make_pair(indexList.at(i), pos3D));
		} else {
			point3DList.insert(std::make_pair(indexList.at(i), Eigen::Vector3d(sf_p[i * 3 + 0], sf_p[i * 3 + 1], sf_p[i * 3 + 2])));
		}
	}

	// 近傍の点を使って補間
	std::map<int, Eigen::Vector3d> point3DList_interp = point3DList;
	for (int i = 0; i < indexList.size(); i++) {
		if (point2DList.at(refImage).find(indexList.at(i)) == point2DList.at(refImage).end() ||
			point3DList_interp.find(indexList.at(i)) != point3DList_interp.end()) {
			continue;
		}

		Eigen::Vector2d pos2D = point2DList.at(refImage).at(indexList.at(i));
		double aveDepth = 0.0;
		double sum = 0;
		for (const auto& j : indexList) {
			if (point3DList.find(j) == point3DList.end() ||
				(pos2D - point2DList.at(refImage).at(j)).norm() > range_interp) {
				continue;
			}
			double weight = (range_interp - (pos2D - point2DList.at(refImage).at(j)).norm());
			aveDepth += point3DList.at(j).z() * weight;
			sum += weight;
		}
		if (sum > 0) {
			aveDepth /= sum;
			plane<double> pl(Eigen::Vector3d(rs_n[refImage * 2 + 0], rs_n[refImage * 2 + 1], 1).normalized(), rs_d[refImage]);
			auto ray = backward(pos2D, pl, cam);
			point3DList_interp.insert(std::make_pair(indexList.at(i), ray.origin + ray.dir * (aveDepth - ray.origin.z()) / ray.dir.z()));
		}
	}

	//
	// 結果出力
	//

	// 初期形状の出力 (デバッグ用)
/*	for (int i = 0; i < numImages; i++) {
		std::ofstream ofs("init" + std::to_string(i) + ".ply");
		writeHeader(ofs, point3DList_init.at(i).size(), false);
		for (const auto& it : point3DList_init.at(i)) {
			writePoint(ofs, it.second);
		}
	}*/

	std::vector<Eigen::Vector3d> camPosList_init, camPosList_result;
	for (int i = 0; i < numImages; i++) {
		camPosList_init.emplace_back(rquats_init.at(i).inverse() * (-tvecs_init.at(i)));
		camPosList_init.emplace_back(rquats_init.at(i).inverse() * (Eigen::Vector3d(0, 0, 0.05) - tvecs_init.at(i)));

		camPosList_result.emplace_back(rquats.at(i).inverse() * (-tvecs.at(i)));
		camPosList_result.emplace_back(rquats.at(i).inverse() * (Eigen::Vector3d(0, 0, 0.05) - tvecs.at(i)));
	}

	std::ofstream ofs_cam_init("points_cam_init.ply");
	writeHeader(ofs_cam_init, camPosList_init.size(), false, camPosList_init.size() / 2);
	writePoints(ofs_cam_init, camPosList_init);
	std::vector<std::pair<int, int>> camEdge_init;
	for (int i = 0; i < camPosList_init.size() / 2; i++) {
		camEdge_init.emplace_back(std::make_pair(i * 2, i * 2 + 1));
	}
	writeEdges(ofs_cam_init, camEdge_init);

	std::ofstream ofs_cam_result("points_cam_result.ply");
	writeHeader(ofs_cam_result, camPosList_result.size(), false, camPosList_result.size() / 2);
	writePoints(ofs_cam_result, camPosList_result);
	std::vector<std::pair<int, int>> camEdge_result;
	for (int i = 0; i < camPosList_result.size() / 2; i++) {
		camEdge_result.emplace_back(std::make_pair(i * 2, i * 2 + 1));
	}
	writeEdges(ofs_cam_result, camEdge_result);

	std::ofstream ofs_points_init("points_init.ply");
	std::ofstream ofs_points_result("points_result.ply");
	std::ofstream ofs_points_result_interp("points_result_interp.ply");

	if (useTexture) {
		auto writeData = [&] (std::ofstream& ofs, const std::map<int, Eigen::Vector3d>& data) {
			writeHeader(ofs, data.size(), true);
			for (const auto& it : data) {
				Eigen::Vector3d color(0, 0, 0);
				if (point2DList.at(refImage).find(it.first) != point2DList.at(refImage).end()) {
					auto pos2D = point2DList.at(refImage).at(it.first);
					auto color_cv = tex.at<cv::Vec3b>(pos2D.y(), pos2D.x());
					color.x() = color_cv[2];
					color.y() = color_cv[1];
					color.z() = color_cv[0];
				}
				writePoint(ofs, it.second, color);
			}
		};

		writeData(ofs_points_init, point3DList_init_ave);
		writeData(ofs_points_result, point3DList);
		writeData(ofs_points_result_interp, point3DList_interp);
	} else {
		std::vector<Eigen::Vector3d> points_init, points_result, points_result_interp;

		for (const auto& it : point3DList_init_ave) {
			points_init.emplace_back(it.second);
		}
		for (const auto& it : point3DList) {
			points_result.emplace_back(it.second);
		}
		for (const auto& it : point3DList_interp) {
			points_result_interp.emplace_back(it.second);
		}

		writePly(ofs_points_init, points_init);
		writePly(ofs_points_result, points_result);
		writePly(ofs_points_result_interp, points_result_interp);
	}

	std::ofstream ofs_nd("nd.ply");
	std::vector<Eigen::Vector3d> nd_points;
	std::vector<std::vector<int>> nd_planes;
	const double planeSize = 0.1;
	for (int i = 0; i < nd.size(); i++) {
		const auto center = nd.at(i).first * nd.at(i).second;
		const auto rightVec = nd.at(i).first.cross(Eigen::Vector3d(0, 1, 0)).normalized();
		const auto upVec = nd.at(i).first.cross(Eigen::Vector3d(1, 0, 0)).normalized();
		nd_points.emplace_back(rquats_init.at(i).inverse() * (center + (rightVec * planeSize + upVec * planeSize) - tvecs_init.at(i)));
		nd_points.emplace_back(rquats_init.at(i).inverse() * (center + (rightVec * planeSize - upVec * planeSize) - tvecs_init.at(i)));
		nd_points.emplace_back(rquats_init.at(i).inverse() * (center + (-rightVec * planeSize - upVec * planeSize) - tvecs_init.at(i)));
		nd_points.emplace_back(rquats_init.at(i).inverse() * (center + (-rightVec * planeSize + upVec * planeSize) - tvecs_init.at(i)));

		std::vector<int> plane1;
		plane1.push_back(i * 4);
		plane1.push_back(i * 4 + 1);
		plane1.push_back(i * 4 + 2);
		std::vector<int> plane2;
		plane2.push_back(i * 4);
		plane2.push_back(i * 4 + 2);
		plane2.push_back(i * 4 + 3);
		nd_planes.emplace_back(plane1);
		nd_planes.emplace_back(plane2);
	}

	writeHeader(ofs_nd, nd_points.size(), false, 0, nd_planes.size());
	writePoints(ofs_nd, nd_points);
	writeFaces(ofs_nd, nd_planes);

	// データ書き出し
	std::ofstream ofs_nd_txt("nd.txt");
	for (const auto& it : nd) {
		ofs_nd_txt << it.first.x() << " " << it.first.y() << " " << it.first.z() << " " << it.second << std::endl;
	}

	// Soft Constraintを使った場合, 法線を出力
	if (use_softConst) {
		for (int i = 0; i < numImages; i++) {
			std::ofstream ofs_nd_txt("nd" + std::to_string(i) + ".txt");
			for (int j = 0; j < indexList.size(); j++) {
				if (point2DList.at(i).find(indexList.at(j)) == point2DList.at(i).end() ||
					countNumObservedImages(indexList.at(j), point2DList) < 2) {
					continue;
				}

				auto pos2D = point2DList.at(i).at(indexList.at(j));
				ofs_nd_txt << pos2D.x() << " " << pos2D.y() << " " << sf_n[(i * indexList.size() + j) * 2 + 0] << " " << sf_n[(i * indexList.size() + j) * 2 + 1] << std::endl;
			}
		}
	}

	//
	// 後処理
	//
	delete[] cam_r;
	delete[] cam_t;
	delete[] d_p;
	delete[] sf_p;
	delete[] rs_n;
	delete[] rs_d;
	delete[] sf_n;
}