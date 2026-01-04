/**
 * @file base.h
 * @brief Base type definitions and utility macros for the machine learning library
 * 
 * This file provides foundational type aliases and utility macros used throughout
 * the machine learning library. It establishes consistent naming conventions and
 * provides platform-independent types for improved portability.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

/* Integer type aliases for clarity and consistency */
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

/* Boolean type aliases */
typedef i8 b8;
typedef i32 b32;

/* Floating point type alias */
typedef float f32;

/**
 * @brief Convert kilobytes to bytes
 * @param n Number of kilobytes
 * @return Number of bytes (n * 1024)
 */
#define KiB(n) ((u64)(n) << 10)

/**
 * @brief Convert megabytes to bytes
 * @param n Number of megabytes
 * @return Number of bytes (n * 1024 * 1024)
 */
#define MiB(n) ((u64)(n) << 20)

/**
 * @brief Convert gigabytes to bytes
 * @param n Number of gigabytes
 * @return Number of bytes (n * 1024 * 1024 * 1024)
 */
#define GiB(n) ((u64)(n) << 30)

/**
 * @brief Return the minimum of two values
 * @param a First value
 * @param b Second value
 * @return The smaller of a and b
 */
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/**
 * @brief Return the maximum of two values
 * @param a First value
 * @param b Second value
 * @return The larger of a and b
 */
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/**
 * @brief Align a value up to the nearest power-of-2 boundary
 * @param n Value to align
 * @param p Power-of-2 boundary
 * @return n aligned up to the nearest multiple of p
 */
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))


