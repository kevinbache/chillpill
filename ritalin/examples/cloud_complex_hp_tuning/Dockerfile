# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install the nightly version of tensorflow
FROM tensorflow/tensorflow:nightly-py3
WORKDIR /root

# Install curl
RUN apt-get update; apt-get install curl -y

# Copies the trainer code to the docker image.
COPY  . .
RUN pip install --editable .

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "ritalin/examples/cloud_complex_hp_tuning/train.py"]
