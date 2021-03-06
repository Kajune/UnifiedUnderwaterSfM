#pragma once

namespace std {
	template <typename T> inline
	std::complex<T> sqrtc(const std::complex<T>& left) {
		if (left.imag() > T(0)) {
			return std::complex<T>(sqrt((left.real() + sqrt(left.real() * left.real() + left.imag() * left.imag())) / T(2)), 
				sqrt((-left.real() + sqrt(left.real() * left.real() + left.imag() * left.imag())) / T(2)));
		} else {
			return std::complex<T>(sqrt((left.real() + sqrt(left.real() * left.real() + left.imag() * left.imag())) / T(2)),
				-sqrt((-left.real() + sqrt(left.real() * left.real() + left.imag() * left.imag())) / T(2)));
		}
	}

	template <typename T> inline
	T absc(const std::complex<T>& left) {
		return sqrt(left.real() * left.real() + left.imag() * left.imag());
	}

	template <typename T> inline
		std::complex<T> logc(const std::complex<T>& left) {	// return log(complex)
		T arg = atan2(left.imag(), left.real());
		return std::complex<T>(log(absc(left)), arg);
	}

	template <typename T> inline
	std::complex<T> expc(const std::complex<T>& left) {	// return exp(complex)
		return (exp(left.real()) * std::complex<T>(cos(left.imag()), sin(left.imag())));
	}

	template <typename T> inline
	std::complex<T> powc(const std::complex<T>& left, const T& right) {
		if (left.imag() == T(0))
			return std::complex<T>(pow(left.real(), right));
		else
			return (expc(right * logc(left)));
	}
}
