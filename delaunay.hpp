#pragma once
#include "EigenSupport.h"
#include <vector>
#include <list>
#include <stack>

struct Triangle {
	int vertex[3];
};

bool isInTriangle(const Triangle& tri, const Eigen::Vector2d& p, const std::vector<Eigen::Vector2d>& pointList) {
	Eigen::Vector2d ab = pointList.at(tri.vertex[1]) - pointList.at(tri.vertex[0]);
	Eigen::Vector2d bp = p - pointList.at(tri.vertex[1]);
	Eigen::Vector2d bc = pointList.at(tri.vertex[2]) - pointList.at(tri.vertex[1]);
	Eigen::Vector2d cp = p - pointList.at(tri.vertex[2]);
	Eigen::Vector2d ca = pointList.at(tri.vertex[0]) - pointList.at(tri.vertex[2]);
	Eigen::Vector2d ap = p - pointList.at(tri.vertex[0]);

	double c1 = ab.x() * bp.y() - ab.y() * bp.x();
	double c2 = bc.x() * cp.y() - bc.y() * cp.x();
	double c3 = ca.x() * ap.y() - ca.y() * ap.x();

	if ((c1 >= 0 && c2 >= 0 && c3 >= 0) || (c1 <= 0 && c2 <= 0 && c3 <= 0)) {
		return true;
	}

	return false;
}

bool isInCircumscribedCircle(const Triangle& tri, const Eigen::Vector2d& p, const std::vector<Eigen::Vector2d>& pointList) {
	Eigen::Vector2d center;
	double x1 = pointList.at(tri.vertex[0]).x(), x2 = pointList.at(tri.vertex[1]).x(), x3 = pointList.at(tri.vertex[2]).x();
	double y1 = pointList.at(tri.vertex[0]).y(), y2 = pointList.at(tri.vertex[1]).y(), y3 = pointList.at(tri.vertex[2]).y();
	double c = 2.0 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1));
	center.x() = ((y3 - y1) * (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1) + (y1 - y2) * (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1)) / c;
	center.y() = ((x1 - x3) * (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1) + (x2 - x1) * (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1)) / c;

	return (pointList.at(tri.vertex[0]) - center).norm() > (p - center).norm();
}

std::vector<Triangle> triangulation(const std::vector<Eigen::Vector2d>& pointList) {
	std::vector<Eigen::Vector2d> pointList_tmp;
	pointList_tmp.assign(pointList.begin(), pointList.end());

	// 重心の計算
	Eigen::Vector2d center(0, 0);
	for (const auto& it : pointList) {
		center += it;
	}
	center /= pointList.size();
	double radius = 0.0;
	for (const auto& it : pointList) {
		radius = std::max(radius, ( it - center ).norm());
	}
	radius += 1.0;

	// 巨大三角形
	pointList_tmp.emplace_back(center.x(), center.y() - 2.0 * radius);
	pointList_tmp.emplace_back(center.x() - sqrt(3.0) * radius, center.y() + radius);
	pointList_tmp.emplace_back(center.x() + sqrt(3.0) * radius, center.y() + radius);

	std::list<Triangle> triangleList;
	triangleList.emplace_back(Triangle{ (int)pointList_tmp.size() - 3, (int)pointList_tmp.size() - 2, (int)pointList_tmp.size() - 1 });

	// 全ての点を順に追加
	int index = 0;
	for (const auto& it : pointList) {
		// どの三角形の内部にあるか
		auto tri = triangleList.end();
		for (auto it2 = triangleList.begin(); it2 != triangleList.end(); it2++) {
			if (isInTriangle(*it2, it, pointList_tmp)) {
				tri = it2;
				break;
			}
		}
		assert(tri != triangleList.end(), "Strange triangle distribution.");

		// 新しい三角形の追加
		std::stack<std::pair<int, int>> edgeStack;
		for (int i = 0; i < 3; i++) {
			triangleList.emplace_back(Triangle{ index, tri->vertex[i], tri->vertex[(i + 1) % 3] });
			edgeStack.emplace(std::make_pair(tri->vertex[i], tri->vertex[(i + 1) % 3]));
		}
		triangleList.erase(tri);

		// フリッピング
		while (!edgeStack.empty()) {
			const auto edge = edgeStack.top();
			edgeStack.pop();

			// edgeを辺に持つ三角形を探す
			std::vector<std::list<Triangle>::iterator> shareTriangles;
			for (auto it2 = triangleList.begin(); it2 != triangleList.end(); it2++) {
				int count = 0;
				for (int i = 0; i < 3; i++) {
					if (it2->vertex[i] == edge.first || it2->vertex[i] == edge.second) {
						count++;
					}
				}
				if (count == 2) {
					shareTriangles.emplace_back(it2);
				}
				assert(count <= 2, "Strange triangle exists.");
			}

			// 共有三角形が無ければスキップ
			if (shareTriangles.size() <= 1) {
				continue;
			}
			assert(shareTriangles.size() <= 2, "Strange triangle sharing.");

			// 共有されていない頂点を探す
			int p[2] = { -1 };
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 3; j++) {
					if (shareTriangles.at(i)->vertex[j] != edge.first && shareTriangles.at(i)->vertex[j] != edge.second) {
						p[i] = shareTriangles.at(i)->vertex[j];
					}
				}
				assert(p[i] >= 0, "Strange triangle vertex.");
			}

			// 外接円内部にあるかを調べる
			bool needFlip = false;
			for (int i = 0; i < 2; i++) {
				if (isInCircumscribedCircle(*shareTriangles.at(i), pointList_tmp.at(p[(i + 1) % 2]), pointList_tmp)) {
					needFlip = true;
					break;
				}
			}

			if (!needFlip) {
				continue;
			}

			// フリッピング
			for (int i = 0; i < 2; i++) {
				triangleList.erase(shareTriangles.at(i));
			}
			triangleList.emplace_back(Triangle{ p[0], p[1], edge.first });
			triangleList.emplace_back(Triangle{ p[0], p[1], edge.second });
			for (int i = 0; i < 2; i++) {
				edgeStack.emplace(std::make_pair(p[i], edge.first));
				edgeStack.emplace(std::make_pair(p[i], edge.second));
			}
		}

		index++;
	}

	// 巨大三角形の除去
	std::vector<Triangle> triangles;
	for (const auto& it : triangleList) {
		bool contain = false;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				contain = contain || it.vertex[i] == pointList_tmp.size() - (j + 1);
			}
		}
		if (!contain) {
			triangles.emplace_back(it);
		}
	}

	return triangles;
}
