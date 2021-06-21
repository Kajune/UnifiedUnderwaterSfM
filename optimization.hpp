#pragma once

#include "ceres\ceres.h"
#include "lightPath.hpp"

#include <cmath>

template <typename T>
using FUNC = std::function<T(const T&, const T&)>;

template <typename T = double>
struct CrossPoint {
	bool operator()(const T* const t, T* residuals) const {
		Eigen::Vector3t<T> pos3d = Eigen::Vector3t<T>(r.origin) + Eigen::Vector3t<T>(r.dir) * t[0];
		residuals[0] = pos3d.z() - f(pos3d.x(), pos3d.y());
		return true;
	}

	ray<T> r;
	FUNC<T> f;
	CrossPoint(const ray<T>& r, const FUNC<T>& f) : r(r), f(f) {}

	static ceres::CostFunction* Create(const ray<T>& r, const FUNC<T>& f) {
		return new ceres::NumericDiffCostFunction<CrossPoint, ceres::CENTRAL, 1, 1>(new CrossPoint(r, f));
	}
};

template <typename T = double>
struct surface {
	FUNC<T> f, dfdx, dfdy;
	surface(const FUNC<T>& f, const FUNC<T>& dfdx, const FUNC<T>& dfdy)
		: f(f), dfdx(dfdx), dfdy(dfdy) {
	}

	bool getCrossPoint(const ray<T>& in, Eigen::Vector3t<T>& out) const {
		T t(0);
		ceres::Problem problem;
		problem.AddResidualBlock(CrossPoint<T>::Create(in, f), nullptr, &t);
		problem.SetParameterLowerBound(&t, 0, 0);
		ceres::Solver::Options option;
		ceres::Solver::Summary summary;
		ceres::Solve(option, &problem, &summary);
		if (summary.termination_type == ceres::TerminationType::FAILURE) {
			out = in.at(0);
			return false;
		}
		out = in.at(t);
		return true;
	}

	Eigen::Vector3t<T> getNormal(const Eigen::Vector3t<T>& pos) const {
		return Eigen::Vector3t<T>(-dfdx(pos.x(), pos.y()), -dfdy(pos.x(), pos.y()), T(1)).normalized();
	}
};

template <typename T>
struct IterativeForwardFunctor {
	bool operator()(const T* const pos2d, T* residuals) const {
		Eigen::Vector2t<T> pos2d_t(pos2d[0], pos2d[1]);
		auto ray = cam.castRay(pos2d_t);
		if (!rs.getCrossPoint(ray, ray.origin)) {
			return false;
		}

		ray.dir = refract(ray.dir, rs.getNormal(ray.origin), r_w<T>);

		residuals[0] = lineToPointDistance(ray, Eigen::Vector3t<T>(pos3d));

		return true;
	}

	const Eigen::Vector3d& pos3d;
	const surface<T>& rs;
	const camera& cam;

	IterativeForwardFunctor(const Eigen::Vector3d& pos3d, const surface<T>& rs, const camera& cam) : pos3d(pos3d), rs(rs), cam(cam) {}

	static ceres::CostFunction* Create(const Eigen::Vector3d& pos3d, const surface<T>& rs, const camera& cam) {
		return new ceres::NumericDiffCostFunction<IterativeForwardFunctor, ceres::CENTRAL, 1, 2>(new IterativeForwardFunctor(pos3d, rs, cam));
	}
};

template <typename T>
bool iterativeForward(const Eigen::Vector3d& pos3d, const surface<T>& rs, const camera& cam, Eigen::Vector2d& pos2d) {
	double pos2d_[2] = { pos2d.x(), pos2d.y() };
	ceres::Problem problem;
	ceres::CostFunction* cost_function = IterativeForwardFunctor<T>::Create(pos3d, rs, cam);
	problem.AddResidualBlock(cost_function, nullptr, pos2d_);

	ceres::Solver::Options option;
//	option.function_tolerance = 1e-10;
//	option.gradient_tolerance = 1e-16;
//	option.parameter_tolerance = 1e-16;
	ceres::Solver::Summary summary;
	ceres::Solve(option, &problem, &summary);

	if (summary.termination_type == ceres::TerminationType::FAILURE) {
		return false;
	}

	pos2d.x() = pos2d_[0];
	pos2d.y() = pos2d_[1];
	return true;
}

struct NormalSfMFunctor {
	template <typename T> bool operator()(const T* const r, const T* const t, const T* const p, T* residual) const {
		cv::Mat rvec = (cv::Mat_<T>(3, 1) << r[0], r[1], r[2]);
		Eigen::Matrix3t<T> rmat;
		cv::Mat rmat_;
		cv::Rodrigues(rvec, rmat_);
		cv::cv2eigen(rmat_, rmat);

		Eigen::Vector3t<T> pos(p[0], p[1], p[2]), tvec(t[0], t[1], t[2]);
		pos = rmat * pos + tvec;

		Eigen::Vector2t<T> res = cam.project(pos) - Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y()));

		residual[0] = res.x();
		residual[1] = res.y();

		return true;
	}

	Eigen::Vector2d point2D;
	const camera& cam;

	NormalSfMFunctor(const Eigen::Vector2d& point2D_, const camera& cam_)
		: point2D(point2D_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<NormalSfMFunctor, ceres::CENTRAL, 2, 3, 3, 3>(new NormalSfMFunctor(point2D, cam)));
	}
};

struct HardForwardFunctor {
	template <typename T> bool operator()(const T* const r, const T* const t, const T* const p, const T* const n, const T* const d, T* residual) const {
		plane<T> pl(Eigen::Vector3t<T>{ T(n[0]), T(n[1]), T(1) }, d[0]);

		cv::Mat rvec = (cv::Mat_<T>(3, 1) << r[0], r[1], r[2]);
		Eigen::Matrix3t<T> rmat;
		cv::Mat rmat_;
		cv::Rodrigues(rvec, rmat_);
		cv::cv2eigen(rmat_, rmat);

		Eigen::Vector3t<T> pos(p[0], p[1], p[2]), tvec(t[0], t[1], t[2]);
		pos = rmat * pos + tvec;

		Eigen::Vector2t<T> res = cam.project(forward(pos, pl)) - Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y()));

		residual[0] = res.x();
		residual[1] = res.y();

		return true;
	}

	Eigen::Vector2d point2D;
	const camera& cam;

	HardForwardFunctor(const Eigen::Vector2d& point2D_, const camera& cam_)
		: point2D(point2D_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<HardForwardFunctor, ceres::CENTRAL, 2, 3, 3, 3, 2, 1>(new HardForwardFunctor(point2D, cam)));
	}
};

struct SoftForwardFunctor {
	template <typename T> bool operator()(const T* const r, const T* const t, const T* const p, const T* const n, const T* const d, const T* const p_n, T* residual) const {
		plane<T> pl(Eigen::Vector3t<T>{ T(n[0]), T(n[1]), T(1) }, d[0]);

		cv::Mat rvec = (cv::Mat_<T>(3, 1) << r[0], r[1], r[2]);
		Eigen::Matrix3t<T> rmat;
		cv::Mat rmat_;
		cv::Rodrigues(rvec, rmat_);
		cv::cv2eigen(rmat_, rmat);

		Eigen::Vector3t<T> pos(p[0], p[1], p[2]), tvec(t[0], t[1], t[2]);
		pos = rmat * pos + tvec;

		auto ray = cam.castRay(Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y())));
		ray = intersectPlane(ray, pl);
		ray.dir = refract(ray.dir, (pl.getNormal(ray) + Eigen::Vector3t<T>(p_n[0], p_n[1], T(1))).normalized(), r_w<T>);

		Eigen::Vector3t<T> pos3d = ray.origin + ray.dir * (pos.z() - ray.origin.z()) / ray.dir.z();

		residual[0] = pos3d.x() - pos.x();
		residual[1] = pos3d.y() - pos.y();

		return true;
	}

	Eigen::Vector2d point2D;
	const camera& cam;

	SoftForwardFunctor(const Eigen::Vector2d& point2D_, const camera& cam_)
		: point2D(point2D_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<SoftForwardFunctor, ceres::CENTRAL, 2, 3, 3, 3, 2, 1, 2>(new SoftForwardFunctor(point2D, cam)));
	}
};

struct HardForwardFunctorCamStatic {
	template <typename T> bool operator()(const T* const p, const T* const n, const T* const d, T* residual) const {
		plane<T> pl(Eigen::Vector3t<T>{ T(n[0]), T(n[1]), T(1) }, d[0]);

		Eigen::Vector3t<T> pos(p[0], p[1], p[2]);

		Eigen::Vector2t<T> res = cam.project(forward(pos, pl)) - Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y()));

		residual[0] = res.x();
		residual[1] = res.y();

		return true;
	}

	Eigen::Vector2d point2D;
	const camera& cam;

	HardForwardFunctorCamStatic(const Eigen::Vector2d& point2D_, const camera& cam_)
		: point2D(point2D_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<HardForwardFunctorCamStatic, ceres::CENTRAL, 2, 3, 2, 1>(new HardForwardFunctorCamStatic(point2D, cam)));
	}
};

struct SoftForwardFunctorCamStatic {
	template <typename T> bool operator()(const T* const p, const T* const n, const T* const d, const T* const p_n, T* residual) const {
		plane<T> pl(Eigen::Vector3t<T>{ T(n[0]), T(n[1]), T(1) }, d[0]);

		Eigen::Vector3t<T> pos(p[0], p[1], p[2]);

		auto ray = cam.castRay(Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y())));
		ray = intersectPlane(ray, pl);
		ray.dir = refract(ray.dir, (pl.getNormal(ray) + Eigen::Vector3t<T>(p_n[0], p_n[1], T(1))).normalized(), r_w<T>);

		Eigen::Vector3t<T> pos3d = ray.origin + ray.dir * (pos.z() - ray.origin.z()) / ray.dir.z();

		residual[0] = pos3d.x() - pos.x();
		residual[1] = pos3d.y() - pos.y();

		return true;
	}

	Eigen::Vector2d point2D;
	const camera& cam;

	SoftForwardFunctorCamStatic(const Eigen::Vector2d& point2D_, const camera& cam_)
		: point2D(point2D_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<SoftForwardFunctorCamStatic, ceres::CENTRAL, 2, 3, 2, 1, 2>(new SoftForwardFunctorCamStatic(point2D, cam)));
	}
};


struct HardForwardFunctorDepth {
	template <typename T> bool operator()(const T* const r, const T* const t, const T* const d_p, const T* const n_ref, const T* const d_ref, const T* const n, const T* const d, T* residual) const {
		plane<T> pl_ref(Eigen::Vector3t<T>{ T(n_ref[0]), T(n_ref[1]), T(1) }, d_ref[0]);
		plane<T> pl(Eigen::Vector3t<T>{ T(n[0]), T(n[1]), T(1) }, d[0]);

		auto ray = backward(point2D_ref, pl_ref, cam);
		Eigen::Vector3t<T> pos = ray.origin + ray.dir * d_p[0] / ray.dir.z();

		cv::Mat rvec = (cv::Mat_<T>(3, 1) << r[0], r[1], r[2]);
		Eigen::Matrix3t<T> rmat;
		cv::Mat rmat_;
		cv::Rodrigues(rvec, rmat_);
		cv::cv2eigen(rmat_, rmat);

		Eigen::Vector3t<T> tvec(t[0], t[1], t[2]);
		pos = rmat * pos + tvec;

		Eigen::Vector2t<T> res = cam.project(forward(pos, pl)) - Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y()));

		residual[0] = res.x();
		residual[1] = res.y();

		return true;
	}

	Eigen::Vector2d point2D, point2D_ref;
	const camera& cam;

	HardForwardFunctorDepth(const Eigen::Vector2d& point2D_, const Eigen::Vector2d& point2D_ref_, const camera& cam_)
		: point2D(point2D_), point2D_ref(point2D_ref_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const Eigen::Vector2d& point2D_ref, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<HardForwardFunctorDepth, ceres::CENTRAL, 2, 3, 3, 1, 2, 1, 2, 1>(new HardForwardFunctorDepth(point2D, point2D_ref, cam)));
	}
};

struct NormalSfMFunctorDepth {
	template <typename T> bool operator()(const T* const r, const T* const t, const T* const d_p, T* residual) const {
		auto ray = cam.castRay(point2D_ref);
		Eigen::Vector3t<T> pos = ray.origin + ray.dir * d_p[0] / ray.dir.z();

		cv::Mat rvec = (cv::Mat_<T>(3, 1) << r[0], r[1], r[2]);
		Eigen::Matrix3t<T> rmat;
		cv::Mat rmat_;
		cv::Rodrigues(rvec, rmat_);
		cv::cv2eigen(rmat_, rmat);

		Eigen::Vector3t<T> tvec(t[0], t[1], t[2]);
		pos = rmat * pos + tvec;

		Eigen::Vector2t<T> res = cam.project(pos) - Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y()));

		residual[0] = res.x();
		residual[1] = res.y();

		return true;
	}

	Eigen::Vector2d point2D, point2D_ref;
	const camera& cam;

	NormalSfMFunctorDepth(const Eigen::Vector2d& point2D_, const Eigen::Vector2d& point2D_ref_, const camera& cam_)
		: point2D(point2D_), point2D_ref(point2D_ref_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const Eigen::Vector2d& point2D_ref, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<NormalSfMFunctorDepth, ceres::CENTRAL, 2, 3, 3, 1>(new NormalSfMFunctorDepth(point2D, point2D_ref, cam)));
	}
};

struct HardForwardFunctorDepthRsStatic {
	template <typename T> bool operator()(const T* const r, const T* const t, const T* const d_p, const T* const n, const T* const d, T* residual) const {
		plane<T> pl_ref(Eigen::Vector3t<T>{ T(n[0]), T(n[1]), T(1) }, d[0]);

		auto ray = backward(point2D_ref, pl_ref, cam);
		Eigen::Vector3t<T> pos = ray.origin + ray.dir * d_p[0] / ray.dir.z();

		cv::Mat rvec = (cv::Mat_<T>(3, 1) << r[0], r[1], r[2]);
		Eigen::Matrix3t<T> rmat;
		cv::Mat rmat_;
		cv::Rodrigues(rvec, rmat_);
		cv::cv2eigen(rmat_, rmat);

		Eigen::Vector3t<T> tvec(t[0], t[1], t[2]);
		pos = rmat * pos + tvec;

		Eigen::Vector3t<T> normal = rmat * Eigen::Vector3t<T>(n[0], n[1], T(1)).normalized();
		T depth = (rmat * Eigen::Vector3t<T>(T(0), T(0), d[0]) + tvec).z();
		plane<T> pl(normal, depth);

		Eigen::Vector2t<T> res = cam.project(forward(pos, pl)) - Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y()));

		residual[0] = res.x();
		residual[1] = res.y();

		return true;
	}

	Eigen::Vector2d point2D, point2D_ref;
	const camera& cam;

	HardForwardFunctorDepthRsStatic(const Eigen::Vector2d& point2D_, const Eigen::Vector2d& point2D_ref_, const camera& cam_)
		: point2D(point2D_), point2D_ref(point2D_ref_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const Eigen::Vector2d& point2D_ref, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<HardForwardFunctorDepthRsStatic, ceres::CENTRAL, 2, 3, 3, 1, 2, 1>(new HardForwardFunctorDepthRsStatic(point2D, point2D_ref, cam)));
	}
};

struct HardForwardFunctorDepthCamStatic {
	template <typename T> bool operator()(const T* const d_p, const T* const n_ref, const T* const d_ref, const T* const n, const T* const d, T* residual) const {
		plane<T> pl_ref(Eigen::Vector3t<T>{ T(n_ref[0]), T(n_ref[1]), T(1) }, d_ref[0]);
		plane<T> pl(Eigen::Vector3t<T>{ T(n[0]), T(n[1]), T(1) }, d[0]);

		auto ray = backward(point2D_ref, pl_ref, cam);
		Eigen::Vector3t<T> pos = ray.origin + ray.dir * d_p[0] / ray.dir.z();
		Eigen::Vector2t<T> res = cam.project(forward(pos, pl)) - Eigen::Vector2t<T>(T(point2D.x()), T(point2D.y()));

		residual[0] = res.x();
		residual[1] = res.y();

		return true;
	}

	Eigen::Vector2d point2D, point2D_ref;
	const camera& cam;

	HardForwardFunctorDepthCamStatic(const Eigen::Vector2d& point2D_, const Eigen::Vector2d& point2D_ref_, const camera& cam_)
		: point2D(point2D_), point2D_ref(point2D_ref_), cam(cam_) {
	}

	static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const Eigen::Vector2d& point2D_ref, const camera& cam) {
		return (new ceres::NumericDiffCostFunction<HardForwardFunctorDepthCamStatic, ceres::CENTRAL, 2, 1, 2, 1, 2, 1>(new HardForwardFunctorDepthCamStatic(point2D, point2D_ref, cam)));
	}
};

struct NormalRegularizer {
	template <typename T> bool operator()(const T* const n1, const T* const n2, T* residual) const {
		residual[0] = (T(1) - Eigen::Vector3t<T>(n1[0], n1[1], T(1)).normalized().dot(Eigen::Vector3t<T>(n2[0], n2[1], T(1)).normalized())) * T(lambda);
		return true;
	}

	double lambda;

	NormalRegularizer(double lambda) : lambda(lambda) {}

	static ceres::CostFunction* Create(double lambda) {
		return (new ceres::AutoDiffCostFunction<NormalRegularizer, 1, 2, 2>(new NormalRegularizer(lambda)));
	}
};
