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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class MLP_Encoder(nn.Module):
    is_mlp_encoder = True
    is_vae = False

    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dims=[256, 256],
        activation="elu",
        orthogonal_init=False,
        output_detach=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(MLP_Encoder, self).__init__()

        self.orthogonal_init = orthogonal_init
        self.output_detach = output_detach
        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim

        activation = get_activation(activation)

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_input_dim, hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], num_output_dim))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        print(f"Encoder MLP: {self.encoder}")

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def forward(self, input):
        return self.encoder(input)

    def encode(self, input):
        self.encoder_out = self.encoder(input)
        if self.output_detach:
            return self.encoder_out.detach()
        else:
            return self.encoder_out

    def get_encoder_out(self):
        return self.encoder_out

    def inference(self, input):
        with torch.no_grad():
            return self.encoder(input)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
