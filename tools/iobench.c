/* iobench.c -- threaded random-pread SSD microbenchmark.
 *
 * Modeled on colibri's iobench (JustVugg/colibri, Apache-2.0; see NOTICE).
 * Measures SEPIA's actual decode access pattern: N threads issue random
 * preads of a fixed size against one file, buffered or F_NOCACHE (macOS
 * only -- this tool has no other target).
 *
 * Usage: ./iobench <file> <block_mb> <n_reads> <threads> <direct 0|1> [align_bytes]
 *   block_mb    : read size in MiB
 *   n_reads     : total preads across all threads (split as evenly as possible)
 *   threads     : worker thread count
 *   direct      : 0 = buffered (page cache), 1 = F_NOCACHE
 *   align_bytes : optional, default 0 = offsets are block-aligned (a multiple
 *                 of block_bytes, the original behavior: task 0.5's ssd-bench
 *                 matrix). A positive value (e.g. 32, GGUF's tensor-data
 *                 alignment) instead draws offsets uniformly at random from
 *                 every align_bytes-aligned position that keeps the whole
 *                 block inside the file -- the task 0.7 unaligned-offset A/B.
 *
 * Reports total GB read (decimal, bytes/1e9), wall seconds, aggregate GB/s,
 * mean ms/block, and p99 ms/block over all reads.
 */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#define ALIGN 16384u

typedef struct {
    int fd;
    size_t block_bytes;
    long n_reads;
    long num_positions; /* count of candidate offsets, spaced step_bytes apart */
    size_t step_bytes;  /* block_bytes (block-aligned) or align_bytes (unaligned) */
    double *lat_ms;      /* this thread's slice of the shared latency array */
} worker_arg;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void *worker(void *argp) {
    worker_arg *a = argp;
    void *buf;
    if (posix_memalign(&buf, ALIGN, a->block_bytes) != 0) {
        fprintf(stderr, "iobench: posix_memalign failed\n");
        exit(1);
    }
    for (long i = 0; i < a->n_reads; i++) {
        long idx = (long)arc4random_uniform((uint32_t)a->num_positions);
        off_t offset = (off_t)idx * (off_t)a->step_bytes;
        double t0 = now_ms();
        ssize_t r = pread(a->fd, buf, a->block_bytes, offset);
        double t1 = now_ms();
        if (r < 0) {
            fprintf(stderr, "iobench: pread failed at offset %lld: %s\n",
                    (long long)offset, strerror(errno));
            exit(1);
        }
        if ((size_t)r != a->block_bytes) {
            fprintf(stderr,
                    "iobench: short read at offset %lld: got %zd, wanted %zu\n",
                    (long long)offset, r, a->block_bytes);
            exit(1);
        }
        a->lat_ms[i] = t1 - t0;
    }
    free(buf);
    return NULL;
}

static int cmp_double(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr,
                "usage: %s <file> <block_mb> <n_reads> <threads> <direct 0|1> "
                "[align_bytes]\n",
                argv[0]);
        return 2;
    }
    const char *path = argv[1];
    long block_mb = strtol(argv[2], NULL, 10);
    long n_reads = strtol(argv[3], NULL, 10);
    long threads = strtol(argv[4], NULL, 10);
    int direct = (int)strtol(argv[5], NULL, 10);
    long align_bytes = (argc == 7) ? strtol(argv[6], NULL, 10) : 0;
    if (block_mb <= 0 || n_reads <= 0 || threads <= 0 ||
        (direct != 0 && direct != 1) || align_bytes < 0) {
        fprintf(stderr, "iobench: bad arguments\n");
        return 2;
    }
    size_t block_bytes = (size_t)block_mb * 1024u * 1024u;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "iobench: open(%s): %s\n", path, strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "iobench: fstat: %s\n", strerror(errno));
        return 1;
    }
    if ((size_t)st.st_size < block_bytes) {
        fprintf(stderr,
                "iobench: file too small (%lld bytes) for %ld MiB blocks\n",
                (long long)st.st_size, block_mb);
        return 1;
    }
    /* Candidate offsets: block-aligned (step = block_bytes, the original
     * behavior) or, with align_bytes > 0, every align_bytes-aligned offset
     * that still keeps the whole block inside the file. */
    long num_positions;
    size_t step_bytes;
    if (align_bytes == 0) {
        step_bytes = block_bytes;
        num_positions = (long)((size_t)st.st_size / step_bytes);
    } else {
        step_bytes = (size_t)align_bytes;
        off_t max_start = (off_t)st.st_size - (off_t)block_bytes;
        num_positions = (long)(max_start / (off_t)step_bytes) + 1;
    }
    if (num_positions < 1 || (unsigned long)num_positions > UINT32_MAX) {
        fprintf(stderr, "iobench: invalid offset range (%ld positions)\n",
                num_positions);
        return 1;
    }
    if (direct) {
        if (fcntl(fd, F_NOCACHE, 1) != 0) {
            fprintf(stderr, "iobench: fcntl(F_NOCACHE): %s\n", strerror(errno));
            return 1;
        }
    }

    double *lat_ms = malloc((size_t)n_reads * sizeof(double));
    if (!lat_ms) {
        fprintf(stderr, "iobench: out of memory for %ld latency samples\n",
                n_reads);
        return 1;
    }

    pthread_t *tids = malloc((size_t)threads * sizeof(pthread_t));
    worker_arg *args = malloc((size_t)threads * sizeof(worker_arg));
    long base = n_reads / threads, rem = n_reads % threads;
    long off = 0;
    for (long t = 0; t < threads; t++) {
        long count = base + (t < rem ? 1 : 0);
        args[t] = (worker_arg){.fd = fd,
                                .block_bytes = block_bytes,
                                .n_reads = count,
                                .num_positions = num_positions,
                                .step_bytes = step_bytes,
                                .lat_ms = lat_ms + off};
        off += count;
    }

    double t_start = now_ms();
    for (long t = 0; t < threads; t++) {
        if (pthread_create(&tids[t], NULL, worker, &args[t]) != 0) {
            fprintf(stderr, "iobench: pthread_create failed\n");
            return 1;
        }
    }
    for (long t = 0; t < threads; t++)
        pthread_join(tids[t], NULL);
    double t_end = now_ms();

    double wall_s = (t_end - t_start) / 1000.0;
    double total_bytes = (double)n_reads * (double)block_bytes;
    double total_gb = total_bytes / 1e9;
    double gbps = total_gb / wall_s;

    qsort(lat_ms, (size_t)n_reads, sizeof(double), cmp_double);
    double sum = 0;
    for (long i = 0; i < n_reads; i++)
        sum += lat_ms[i];
    double mean_ms = sum / (double)n_reads;
    long p99_idx = (long)(0.99 * (double)(n_reads - 1));
    double p99_ms = lat_ms[p99_idx];

    printf("file=%s block_mb=%ld n_reads=%ld threads=%ld direct=%d "
           "align_bytes=%ld\n",
           path, block_mb, n_reads, threads, direct, align_bytes);
    printf("total_gb=%.4f wall_s=%.4f gbps=%.4f mean_ms=%.4f p99_ms=%.4f\n",
           total_gb, wall_s, gbps, mean_ms, p99_ms);

    free(lat_ms);
    free(tids);
    free(args);
    close(fd);
    return 0;
}
