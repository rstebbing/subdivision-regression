// patch_index_encoding.cpp

// Try not to vomit ...

// Includes
#include <stdint.h>
#include <cassert>

// SetDoubleToFixedPointInUnitInterval
static void SetDoubleToFixedPointInUnitInterval(double* d) {
  uint64_t* ull = (uint64_t*)d;
  uint32_t e = (*ull >> 52) & 0x7FFF;

  // Handle less than 0 or fixed-point precision.
  if ((*ull & 0x8000000000000000) != 0 || e < (1023 - 52)) {
    *ull = 0;
    return;
  }

  // Ensure less than 1.
  if (e >= 1023) {
    *ull = 0x3FEFFFFFFFFFFFFF;
    e = 1022;
  }

  // Clean sign and exponent bits.
  *ull &= 0x000FFFFFFFFFFFFF;
  *ull |= 0x0010000000000000;

  // Set to fixed-point representation.
  while (e++ <= 1022) {
    *ull >>= 1;
  }
}

// SetDoubleFromFixedPointInUnitInterval
static void SetDoubleFromFixedPointInUnitInterval(double* d) {
  uint64_t* ull = (uint64_t*)d;
  // Clear sign bit and reset exponent.
  *ull &= 0x000FFFFFFFFFFFFF;

  // Determine true exponent `e` and set.
  uint64_t e = 1023;
  while (e >= (1023 - 52) && (*ull & 0x0010000000000000) == 0) {
    *ull <<= 1;
    --e;
  }

  *ull &= 0x000FFFFFFFFFFFFF;
  if (e >= (1023 - 52)) {
    *ull |= e << 52;
  }
}

// EncodePatchIndexInPlace
void EncodePatchIndexInPlace(double* u, int patch_index) {
  assert(0 <= patch_index && patch_index < (1 << 24));

  SetDoubleToFixedPointInUnitInterval(&u[0]);
  SetDoubleToFixedPointInUnitInterval(&u[1]);

  *(uint64_t*)&u[0] |= (uint64_t)(patch_index & 0xFFF) << 52;
  *(uint64_t*)&u[1] |= (uint64_t)((patch_index >> 12) & 0xFFF) << 52;
}

// DecodePatchIndexInPlace
int DecodePatchIndexInPlace(double* u) {
  uint32_t lo = (*(uint64_t*)&u[0] >> 52) & 0xFFF;
  uint32_t hi = (*(uint64_t*)&u[1] >> 52) & 0xFFF;
  int patch_index = (hi << 12) | lo;

  SetDoubleFromFixedPointInUnitInterval(&u[0]);
  SetDoubleFromFixedPointInUnitInterval(&u[1]);

  return patch_index;
}
