# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from the cp-precision-debug skill (MIT) for CP1/CP2 precision debugging.

"""CP Debug tool for comparing CP1 vs CP2 training precision.

Records per-layer in/out/weight/grad via hooks, dumps to disk, then compares
CP1 and CP2 dumps offline to locate the first divergent layer.

Usage:
    from llamafactory.v1.core.cp_debug import register_cp_debug_hooks, CPDebugConfig

    config = CPDebugConfig(
        enabled=True,
        mode="dump",
        expected_seq_len=4096,
        cp_group=ps.get_group("cp"),
    )
    manager = register_cp_debug_hooks(model, config)

    # train ...

    # compare
    python -m llamafactory.v1.core.cp_debug.compare ./cp1_dumps ./cp2_dumps
"""

from .hooks import (
    CPDebugConfig,
    CPDebugManager,
    NoOpCPDebugManager,
    register_cp_debug_hooks,
)
from .utils import all_gather_seq, detect_seq_dim, tensor_stats


__all__ = [
    "CPDebugConfig",
    "CPDebugManager",
    "NoOpCPDebugManager",
    "all_gather_seq",
    "detect_seq_dim",
    "register_cp_debug_hooks",
    "tensor_stats",
]
