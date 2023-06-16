
## End to End ML Project



[Check the data source](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4954928053318020/1058911316420443/167703932442645/latest.html)


1. Create environment

`conda create -p venv python==3.8 -y`

2. Activate environment

`conda activate venv/`

2. Install dependencies

`pip install -r requirements.txt`

3. Run the app

`streamlit run app.py`


## Create Docker image

Run the following commands to create and run docker image

1. Create docker image
`docker build -t insurance_claims .`

2. Check image exists 
`docker images`

3. run the image
`docker run -p 5000:5000 insurance_claims`

4. Check the containers
`docker ps`

5. Stop the container
`docker stop container_id`

6. Login to DockerHub
`docker login`

7. Remove a docker image
`docker image rm -f image-name`

8. Rename the image - Part 1
`docker build -t gaurav178829/insurance_claims .` - use your own user_name

9. Push the image into dockerhub repo
`docker push gaurav178829/insurance_claims:latest`

10. Run the docker image in detach mode
`docker run -d -p 5000:5000 gaurav178829/insurance_claims:latest`


Run the commands

`sudo apt-get upgrade`
`curl -fsSL https://get.docker.com -o get-docker.sh`
`sudo sh get-docker.sh`
`sudo usermod -aG docker ubuntu`
`newgrp docker`


AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_ECR_LOGIN_URI
ECR_REPOSITORY_NAME