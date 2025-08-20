# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import (
    LEGGED_GYM_ROOT_DIR,
    LEGGED_GYM_ENVS_DIR,
)
import os, sys
from legged_gym.utils.task_registry import task_registry

robot_type = os.getenv("ROBOT_TYPE")

if not robot_type:
    print("\033[1m\033[31mError: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.\033[0m")
    sys.exit(1)

if robot_type.startswith("PF"):
    if robot_type in ["PF_TRON1A", "PF_P441A", "PF_P441B", "PF_P441C", "PF_P441C2"]:
        from legged_gym.envs.pointfoot_flat.pointfoot_flat import BipedPF
        from legged_gym.envs.pointfoot_flat.pointfoot_flat_config import BipedCfgPF, BipedCfgPPOPF
        task_registry.register("pointfoot_flat", BipedPF, BipedCfgPF(), BipedCfgPPOPF())
    else:
        print("\033[1m\033[31mError: Input ROBOT_TYPE={}".format(robot_type), 
        "is not among valid robot types PF_TRON1A, PF_P441A, PF_P441B, PF_P441C, PF_P441C2.\033[0m")
        sys.exit(1)

elif robot_type == "SF_TRON1A":
    from legged_gym.envs.solefoot_flat.solefoot_flat import BipedSF
    from legged_gym.envs.solefoot_flat.solefoot_flat_config import BipedCfgSF, BipedCfgPPOSF
    task_registry.register("pointfoot_flat", BipedSF, BipedCfgSF(), BipedCfgPPOSF())

elif robot_type == "WF_TRON1A":
    from legged_gym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF
    from legged_gym.envs.wheelfoot_flat.wheelfoot_flat_config import BipedCfgWF, BipedCfgPPOWF
    from legged_gym.envs.wheelfoot_rough.wheelfoot_rough import BipedWF_Rough
    from legged_gym.envs.wheelfoot_rough.wheelfoot_rough_config import BipedCfgWF_Rough, BipedCfgPPOWF_Rough
    task_registry.register("pointfoot_flat", BipedWF, BipedCfgWF(), BipedCfgPPOWF())
    task_registry.register("wheelfoot_rough", BipedWF_Rough, BipedCfgWF_Rough(), BipedCfgPPOWF_Rough())
else:
    print("\033[1m\033[31mError: Input ROBOT_TYPE={}".format(robot_type), 
        "is not among valid robot types PF_P441A, PF_P441B, PF_P441C, PF_P441C2, PF_TRON1A, WF_TRON1A and SF_TRON1A.\033[0m")
    sys.exit(1)