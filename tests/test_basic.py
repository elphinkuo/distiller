#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import distiller

def test_sparsity():
    zeros = torch.zeros(2,3,5,6)
    print(distiller.sparsity(zeros))
    assert distiller.sparsity(zeros) == 1.0

    ones = torch.zeros(12,43,4,6)
    ones.fill_(1)
    assert distiller.sparsity(ones) == 0.0