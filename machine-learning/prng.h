/**
 * @file prng.h
 * @brief Pseudo-random number generator (PCG algorithm)
 * 
 * This file implements the PCG (Permuted Congruential Generator) random number
 * generator, which provides high-quality random numbers with good statistical
 * properties and performance.
 * 
 * Based on PCG Random Number Generator (https://www.pcg-random.org)
 * Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
 * 
 * The PCG algorithm combines a linear congruential generator with a permutation
 * function to produce high-quality randomness suitable for Monte Carlo methods,
 * simulations, and neural network initialization.
 */

/**
 * @brief PCG random number generator state
 * 
 * Encapsulates the internal state of the random number generator.
 * Multiple instances can be used for independent random sequences.
 */
typedef struct {
    u64 state;  /**< Internal RNG state */
    u64 inc;    /**< Stream selector (must be odd) */
} prng_state;

/**
 * @brief Seed a random number generator instance
 * @param rng Pointer to RNG state
 * @param initstate Initial state value (seed)
 * @param initseq Sequence selector for independent streams
 * 
 * Initializes the RNG with the given seed and sequence. Different sequences
 * with the same seed will produce different random number streams.
 */
void prng_seed_r(prng_state* rng, u64 initstate, u64 initseq);

/**
 * @brief Seed the global random number generator
 * @param initstate Initial state value (seed)
 * @param initseq Sequence selector for independent streams
 */
void prng_seed(u64 initstate, u64 initseq);

/**
 * @brief Generate a random 32-bit unsigned integer
 * @param rng Pointer to RNG state
 * @return Random 32-bit unsigned integer
 */
u32 prng_rand_r(prng_state* rng);

/**
 * @brief Generate a random 32-bit unsigned integer (global RNG)
 * @return Random 32-bit unsigned integer
 */
u32 prng_rand(void);

/**
 * @brief Generate a random float in [0, 1)
 * @param rng Pointer to RNG state
 * @return Random float in the range [0, 1)
 */
f32 prng_randf_r(prng_state* rng);

/**
 * @brief Generate a random float in [0, 1) (global RNG)
 * @return Random float in the range [0, 1)
 */
f32 prng_randf(void);


