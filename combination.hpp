#pragma once

// possible implementation introduced at http://en.cppreference.com/w/cpp/algorithm/rotate with slight modification to handle parted ranges
template<typename FI>
void parted_rotate(FI first1, FI last1, FI first2, FI last2) {
	if (first1 == last1 || first2 == last2) return;
	FI next = first2;
	while (first1 != next) {
		std::iter_swap(first1++, next++);
		if (first1 == last1) first1 = first2;
		if (next == last2) {
			next = first2;
		} else if (first1 == first2) {
			first2 = next;
		}
	}
}

template<typename BI>
bool next_combination_imp(BI first1, BI last1, BI first2, BI last2) {
	if (first1 == last1 || first2 == last2) return false;
	auto target = last1; --target;
	auto last_elem = last2; --last_elem;
	// find right-most incrementable element: target
	while (target != first1 && !(*target < *last_elem)) --target;
	if (target == first1 && !(*target < *last_elem)) {
		parted_rotate(first1, last1, first2, last2);
		return false;
	}
	// find the next value to be incremented: *next
	auto next = first2;
	while (!(*target < *next)) ++next;
	std::iter_swap(target++, next++);
	parted_rotate(target, last1, next, last2);
	return true;
}

// INVARIANT: is_sorted(first, mid) && is_sorted(mid, last)
template<typename BI>
inline bool next_combination(BI first, BI mid, BI last) {
	return next_combination_imp(first, mid, mid, last);
}

// INVARIANT: is_sorted(first, mid) && is_sorted(mid, last)
template<typename BI>
inline bool prev_combination(BI first, BI mid, BI last) {
	return next_combination_imp(mid, last, first, mid);
}
