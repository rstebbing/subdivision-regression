////////////////////////////////////////////
// File: patch_index_encoding.h           //
// Copyright Richard Stebbing 2015.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////
#ifndef PATCH_INDEX_ENCODING_H
#define PATCH_INDEX_ENCODING_H

void EncodePatchIndexInPlace(double* u, int patch_index);
int DecodePatchIndexInPlace(double* u);

#endif // PATCH_INDEX_ENCODING_H
