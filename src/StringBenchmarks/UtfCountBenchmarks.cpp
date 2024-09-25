#include <cstddef>
#include <cstdint>
#include <span>

#ifdef __x86_64__
#include <emmintrin.h>  // SSE2.
#endif

#include <benchmark/benchmark.h>

int naive_utf8_length(std::span<const uint8_t> span) {
  int utf8_length = 0;
  for (uint8_t c : span) {
    utf8_length += c >> 7;
  }
  return utf8_length;
 }

const uint8_t lorem_ipsum_bytes[] = {
    'L', 'o', 'r', 'e', 'm', ' ', 'i', 'p', 's', 'u', 'm', ' ', 'd', 'o', 'l', 'o', 'r', ' ',
    's', 'i', 't', ' ', 'a', 'm', 'e', 't', ',', ' ', 'c', 'o', 'n', 's', 'e', 'c', 't', 'e',
    't', 0xfc, 'r', ' ', 'a', 'd', 'i', 'p', 'i', 's', 'c', 'i', 'n', 'g', ' ', 'e', 'l', 'i',
    't', ',', ' ', 's', 'e', 'd', ' ', 'd', 'o', ' ', 'e', 'i', 'u', 's', 'm', 'o', 'd', ' ',
    't', 'e', 'm', 'p', 0xf6, 'r', ' ', 'i', 'n', 'c', 'i', 'd', 'i', 'd', 'u', 'n', 't', ' ',
    'u', 't', ' ', 'l', 'a', 'b', 'o', 'r', 'e', ' ', 'e', 't', ' ', 'd', 'o', 'l', 'o', 'r',
    'e', ' ', 'm', 'a', 'g', 'n', 'a', ' ', 'a', 'l', 'i', 'q', 'u', 'a', '.'
};

std::span<const uint8_t> lorem_ipsum_span(lorem_ipsum_bytes, sizeof(lorem_ipsum_bytes));

static void AutoVectorizedUtf8Length(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  for (auto _ : state) {
    auto length = naive_utf8_length(lorem_ipsum_span);
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(length);
  }
}
// Register the function as a benchmark
BENCHMARK(AutoVectorizedUtf8Length);

#define B0 0, 1, 1, 2,
#define B1 1, 2, 2, 3,
#define B2 2, 3, 3, 4,
#define B3 3, 4, 4, 5,
#define B4 4, 5, 5, 6,
#define B5 5, 6, 6, 7,
#define B6 6, 7, 7, 8

#define LINE0 B0 B1 B1 B2
#define LINE1 B1 B2 B2 B3
#define LINE2 B2 B3 B3 B4
#define LINE3 B3 B4 B4 B5
#define LINE4 B4 B5 B5 B6

static uint8_t constexpr popcount_table[256] = {
  LINE0 LINE1 LINE1 LINE2
  LINE1 LINE2 LINE2 LINE3
  LINE1 LINE2 LINE2 LINE3
  LINE2 LINE3 LINE3 LINE4
};

static inline int SixteenBitPopCount(unsigned input) {
#ifdef __POPCNT__
  return __builtin_popcount(input);
#else
  if (input == 0) return 0;
  return popcount_table[input & 0xff] + popcount_table[input >> 8];
#endif
}

#ifdef __x86_64__
int pure_sse2(std::span<const uint8_t> input) {
  // x64 always has SSE2, so we can assume that.
  int utf8_length = input.size();
  const uint8_t* s = input.data();
  ssize_t length = input.size();
  int last_bits = (uintptr_t)s & 15;
  int alignment_mask = 0xffff << last_bits;
  ssize_t i;
  int bits = 0;
  for (i = -last_bits ; i < length; i += 16) {
    // Load aligned to a 128 bit XMM2 register.
    __m128i raw = *(__m128i*)(s + i);
    // Takes the top bit of each byte and puts it in the corresponding bit of a
    // normal integer.  PMOVMSKB.
    bits = _mm_movemask_epi8(raw) & alignment_mask;
    utf8_length += SixteenBitPopCount(bits);
    alignment_mask = 0xffff;
  }
  // Remove the bits from past the end of the string.
  utf8_length -= SixteenBitPopCount(bits & (0xffff << (16 - (i - length))));
  return utf8_length;
}

static void PureSse2(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  for (auto _ : state) {
    auto length = pure_sse2(lorem_ipsum_span);
    benchmark::DoNotOptimize(length);
  }
}
// Register the function as a benchmark
BENCHMARK(PureSse2);
#endif  // __x86_64__
