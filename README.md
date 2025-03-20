**Assignment Detail** 
1. Kindly login to http://st124783.ml.brain.cs.ait.ac.th to view the web application and model in action.
2. I have changed the target variable and classified into 4 categories of <span style="color:red">Budget/Economy/Premium/Luxury</span>. This classification was done based on price point.
3. Streamlit framework has been used for front end web application and for visualization: Plotly and Seaborn mostly
4. This project uses GitHub Actions with a self-hosted runner to automate Docker image builds and deployment to the ML2023 server. Upon pushing changes to the main branch, the pipeline:
	i.	Builds and Pushes the Docker image to Docker Hub.
	ii.	Connects to ML2023 via SSH using a self-hosted runner.
	iii.Pulls the latest image and restarts the container using docker compose.
5.Project Structure
 📦 ML_AIT_A3
 ┣ 📂 app/                  # Source code for the Streamlit app
 ┣ 📜 Dockerfile            # Defines how the Docker container is built
 ┣ 📜 docker-compose.yaml   # Manages multi-container deployment
 ┣ 📜 docker-compose-deploy.yaml  # Used for deployment on ML2023
 ┣ 📜 ci.yml                # GitHub Actions CI/CD pipeline
 ┣ 📜 README.md             # Project documentation
 ┗ 📜 requirements.txt      # Python dependencies
