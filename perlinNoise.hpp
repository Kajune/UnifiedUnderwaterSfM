#pragma once
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <random>

class PerlinNoise {
private:

	std::int32_t p[512];

	static double Fade(double t) noexcept {
		return t * t * t * (t * (t * 6 - 15) + 10);
	}

	static double dFade(double t) noexcept {
		return 30 * t * t * (t - 1) * (t - 1);
	}

	static double Lerp(double t, double a, double b) noexcept {
		return a + t * (b - a);
	}

	static double Grad(std::int32_t hash, double x, double y, double z) noexcept {
		const std::int32_t h = hash & 15;
		const double u = h < 8 ? x : y;
		const double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
		return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
	}

public:

	explicit PerlinNoise(std::uint32_t seed = std::default_random_engine::default_seed) {
		reseed(seed);
	}

	void reseed(std::uint32_t seed) {
		for (size_t i = 0; i < 256; ++i) {
			p[i] = i;
		}

		std::shuffle(std::begin(p), std::begin(p) + 256, std::default_random_engine(seed));

		for (size_t i = 0; i < 256; ++i) {
			p[256 + i] = p[i];
		}
	}

	double noise(double x, double* dx) const {
		return noise(x, 0.0, 0.0, dx, nullptr, nullptr);
	}

	double noise(double x, double y, double* dx, double* dy) const {
		return noise(x, y, 0.0, dx, dy, nullptr);
	}

	double noise(double x, double y, double z, double* dx, double* dy, double* dz) const {
		const std::int32_t X = static_cast<std::int32_t>(std::floor(x)) & 255;
		const std::int32_t Y = static_cast<std::int32_t>(std::floor(y)) & 255;
		const std::int32_t Z = static_cast<std::int32_t>(std::floor(z)) & 255;

		x -= std::floor(x);
		y -= std::floor(y);
		z -= std::floor(z);

		const double u = Fade(x);
		const double v = Fade(y);
		const double w = Fade(z);

		const double du = dFade(x);
		const double dv = dFade(y);
		const double dw = dFade(z);

		const std::int32_t A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
		const std::int32_t B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;

		double a = Grad(p[AA], x, y, z);
		double b = Grad(p[BA], x - 1, y, z);
		double c = Grad(p[AB], x, y - 1, z);
		double d = Grad(p[BB], x - 1, y - 1, z);
		double e = Grad(p[AA + 1], x, y, z - 1);
		double f = Grad(p[BA + 1], x - 1, y, z - 1);
		double g = Grad(p[AB + 1], x, y - 1, z - 1);
		double h = Grad(p[BB + 1], x - 1, y - 1, z - 1);

		double k0 = b - a;
		double k1 = c - a;
		double k2 = e - a;
		double k3 = a + d - b - c;
		double k4 = a + f - b - e;
		double k5 = a + g - c - e;
		double k6 = b + c + e + h - a - d - f - g;

		if (dx) {
			*dx = du * (k0 + v * k3 + w * k4 + v * w * k6);
		}
		if (dy) {
			*dy = dv * (k1 + u * k3 + w * k5 + u * w * k6);
		}
		if (dz) {
			*dz = dw * (k2 + u * k4 + v * k5 + u * v * k6);
		}

		return Lerp(w, 
					Lerp(v, 
						 Lerp(u, a, b),
						 Lerp(u, c, d)),
					Lerp(v, 
						 Lerp(u, e, f),
						 Lerp(u, g, h)));
	}

	double octaveNoise(std::int32_t octaves, double x, double* dx = nullptr) const {
		double result = 0.0;
		double amp = 1.0;
		double dx_ = 0.0;
		if (dx) {
			*dx = 0.0;
		}

		for (std::int32_t i = 0; i < octaves; ++i) {
			result += noise(x, &dx_) * amp;
			x *= 2.0;
			amp *= 0.5;
			if (dx) {
				*dx += dx_ * amp;
			}
		}

		return result;
	}

	double octaveNoise(std::int32_t octaves, double x, double y, double* dx = nullptr, double* dy = nullptr) const {
		double result = 0.0;
		double amp = 1.0;
		double dx_ = 0.0, dy_ = 0.0;
		if (dx) {
			*dx = 0.0;
		}
		if (dy) {
			*dy = 0.0;
		}

		for (std::int32_t i = 0; i < octaves; ++i) {
			result += noise(x, y, &dx_, &dy_) * amp;
			x *= 2.0;
			y *= 2.0;
			amp *= 0.5;
			if (dx) {
				*dx += dx_ * amp;
			}
			if (dy) {
				*dy += dy_ * amp;
			}
		}

		return result;
	}

	double octaveNoise(std::int32_t octaves, double x, double y, double z, double* dx = nullptr, double* dy = nullptr, double* dz = nullptr) const {
		double result = 0.0;
		double amp = 1.0;
		double dx_ = 0.0, dy_ = 0.0, dz_ = 0.0;
		if (dx) {
			*dx = 0.0;
		}
		if (dy) {
			*dy = 0.0;
		}
		if (dz) {
			*dz = 0.0;
		}

		for (std::int32_t i = 0; i < octaves; ++i) {
			result += noise(x, y, z, &dx_, &dy_, &dz_) * amp;
			x *= 2.0;
			y *= 2.0;
			z *= 2.0;
			amp *= 0.5;
			if (dx) {
				*dx += dx_ * amp;
			}
			if (dy) {
				*dy += dy_ * amp;
			}
			if (dz) {
				*dz += dz_ * amp;
			}
		}

		return result;
	}

	double noise0_1(double x, double* dx = nullptr) const {
		return noise(x, dx) * 0.5 + 0.5;
	}

	double noise0_1(double x, double y, double* dx = nullptr, double* dy = nullptr) const {
		return noise(x, y, dx, dy) * 0.5 + 0.5;
	}

	double noise0_1(double x, double y, double z, double* dx = nullptr, double* dy = nullptr, double* dz = nullptr) const {
		return noise(x, y, z, dx, dy, dz) * 0.5 + 0.5;
	}

	double octaveNoise0_1(std::int32_t octaves, double x, double* dx = nullptr) const {
		return octaveNoise(x, octaves) * 0.5 + 0.5;
	}

	double octaveNoise0_1(std::int32_t octaves, double x, double y, double* dx = nullptr, double* dy = nullptr) const {
		return octaveNoise(x, y, octaves) * 0.5 + 0.5;
	}

	double octaveNoise0_1(std::int32_t octaves, double x, double y, double z, double* dx = nullptr, double* dy = nullptr, double* dz = nullptr) const {
		return octaveNoise(x, y, z, octaves) * 0.5 + 0.5;
	}
};
