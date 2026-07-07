# Third-Party Notices

This repository vendors selected third-party code under `third_party/` for the
Lychee-FD demo and online serving pipeline. The notices below document the
upstream projects, local integration changes, and license obligations for the
vendored copies.

## Step-Audio2

- Local path: `third_party/Step-Audio2`
- Upstream project: `https://github.com/stepfun-ai/Step-Audio2`
- License: Apache License 2.0
- Local license copy: `third_party/Step-Audio2/LICENSE`

The local copy is a reduced integration subset used by the Lychee-FD token2wav
and realtime serving flow. When updating this subtree, keep the upstream
Apache-2.0 license file with the vendored code and preserve file-level license
headers.

One file in the vendored Step-Audio2 subset,
`third_party/Step-Audio2/flashcosyvoice/modules/hifigan_components/layers.py`,
contains an upstream attribution comment for code adapted from
`https://github.com/EdwardDixon/snake` under the MIT license. The corresponding
MIT license notice is included at
`third_party/Step-Audio2/incl_licenses/snake/LICENSE`. If that code is kept
vendored, retain both the attribution and the license notice when refreshing or
replacing the Step-Audio2 subtree.

## Model Weights

This repository and Docker image do not vendor model weights. Users who download
Step-Audio2 or other model checkpoints must comply with the license terms
published by the corresponding model providers.
