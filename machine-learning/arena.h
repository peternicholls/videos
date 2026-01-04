/**
 * @file arena.h
 * @brief Memory arena allocator for efficient memory management
 * 
 * This file implements a custom memory arena allocator that provides fast,
 * deterministic memory allocation with minimal overhead. Memory arenas are
 * useful for bulk allocations and deallocations, reducing fragmentation and
 * improving cache locality.
 * 
 * Key features:
 * - Virtual memory reservation with on-demand commit
 * - Fast linear allocation with minimal bookkeeping
 * - Temporary allocation scopes for scratch memory
 * - Thread-local scratch arenas to avoid conflicts
 */

/** Base position offset to account for arena header */
#define ARENA_BASE_POS (sizeof(mem_arena))

/** Alignment for all arena allocations (pointer size) */
#define ARENA_ALIGN (sizeof(void*))

/**
 * @brief Memory arena structure for linear allocation
 * 
 * Manages a region of virtual memory with lazy commitment. Memory is reserved
 * upfront but only committed to physical pages as needed.
 */
typedef struct {
    u64 reserve_size;  /**< Total reserved virtual memory size */
    u64 commit_size;   /**< Size of each commit chunk */
    u64 pos;           /**< Current allocation position */
    u64 commit_pos;    /**< Current committed memory position */
} mem_arena;

/**
 * @brief Temporary arena allocation scope
 * 
 * Used to create temporary allocation scopes that can be easily unwound.
 * Useful for scratch allocations within a function or algorithm.
 */
typedef struct {
    mem_arena* arena;  /**< Pointer to the arena */
    u64 start_pos;     /**< Starting position for this scope */
} mem_arena_temp;

/**
 * @brief Create a new memory arena
 * @param reserve_size Total virtual memory to reserve
 * @param commit_size Size of memory to commit at a time
 * @return Pointer to the created arena, or NULL on failure
 * 
 * Reserves a large virtual address space but only commits physical memory
 * in chunks as needed. This allows for efficient memory usage.
 */
mem_arena* arena_create(u64 reserve_size, u64 commit_size);

/**
 * @brief Destroy a memory arena and release all memory
 * @param arena Arena to destroy
 */
void arena_destroy(mem_arena* arena);

/**
 * @brief Allocate memory from the arena
 * @param arena Arena to allocate from
 * @param size Number of bytes to allocate
 * @param non_zero If true, memory is not zeroed; if false, memory is zeroed
 * @return Pointer to allocated memory, or NULL on failure
 * 
 * Allocates size bytes from the arena with proper alignment. If non_zero is
 * false, the returned memory is guaranteed to be zero-initialized.
 */
void* arena_push(mem_arena* arena, u64 size, b32 non_zero);

/**
 * @brief Deallocate the most recently allocated memory
 * @param arena Arena to deallocate from
 * @param size Number of bytes to deallocate
 * 
 * Moves the allocation position back by size bytes. Only safe for LIFO
 * (stack-like) allocation patterns.
 */
void arena_pop(mem_arena* arena, u64 size);

/**
 * @brief Reset arena position to a specific point
 * @param arena Arena to reset
 * @param pos Position to reset to
 */
void arena_pop_to(mem_arena* arena, u64 pos);

/**
 * @brief Clear all allocations from the arena
 * @param arena Arena to clear
 * 
 * Resets the arena to its initial state, effectively freeing all allocations.
 */
void arena_clear(mem_arena* arena);

/**
 * @brief Begin a temporary allocation scope
 * @param arena Arena to create scope in
 * @return Temporary scope handle
 * 
 * Creates a savepoint in the arena that can be restored later.
 */
mem_arena_temp arena_temp_begin(mem_arena* arena);

/**
 * @brief End a temporary allocation scope
 * @param temp Temporary scope to end
 * 
 * Restores the arena to the state it was in when arena_temp_begin was called.
 */
void arena_temp_end(mem_arena_temp temp);

/**
 * @brief Get a thread-local scratch arena
 * @param conflicts Array of arenas to avoid (may be NULL)
 * @param num_conflicts Number of arenas to avoid
 * @return Temporary scope for scratch allocations
 * 
 * Returns one of two thread-local scratch arenas, avoiding any arenas in the
 * conflicts array. Useful for temporary allocations within a function.
 */
mem_arena_temp arena_scratch_get(mem_arena** conflicts, u32 num_conflicts);

/**
 * @brief Release a scratch arena scope
 * @param scratch Scratch scope to release
 */
void arena_scratch_release(mem_arena_temp scratch);

/**
 * @brief Allocate and return a zeroed structure
 * @param arena Arena to allocate from
 * @param T Type of structure to allocate
 * @return Pointer to allocated structure
 */
#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T), false)

/**
 * @brief Allocate a structure (not zeroed)
 * @param arena Arena to allocate from
 * @param T Type of structure to allocate
 * @return Pointer to allocated structure
 */
#define PUSH_STRUCT_NZ(arena, T) (T*)arena_push((arena), sizeof(T), true)

/**
 * @brief Allocate and return a zeroed array
 * @param arena Arena to allocate from
 * @param T Type of array elements
 * @param n Number of elements
 * @return Pointer to allocated array
 */
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), false)

/**
 * @brief Allocate an array (not zeroed)
 * @param arena Arena to allocate from
 * @param T Type of array elements
 * @param n Number of elements
 * @return Pointer to allocated array
 */
#define PUSH_ARRAY_NZ(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), true)

/* Platform-specific memory management functions */

/**
 * @brief Get the system page size
 * @return Page size in bytes
 */
u32 plat_get_pagesize(void);

/**
 * @brief Reserve virtual memory without committing
 * @param size Size to reserve
 * @return Pointer to reserved memory, or NULL on failure
 */
void* plat_mem_reserve(u64 size);

/**
 * @brief Commit reserved memory to physical pages
 * @param ptr Pointer to reserved memory
 * @param size Size to commit
 * @return true on success, false on failure
 */
b32 plat_mem_commit(void* ptr, u64 size);

/**
 * @brief Decommit memory (return physical pages to OS)
 * @param ptr Pointer to committed memory
 * @param size Size to decommit
 * @return true on success, false on failure
 */
b32 plat_mem_decommit(void* ptr, u64 size);

/**
 * @brief Release reserved memory
 * @param ptr Pointer to reserved memory
 * @param size Size to release
 * @return true on success, false on failure
 */
b32 plat_mem_release(void* ptr, u64 size);


