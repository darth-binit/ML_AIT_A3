**Assignment Detail** 
1. Kindly login to http://st124783.ml.brain.cs.ait.ac.th to view the web application and model in action.
2. I have changed the target variable and classified into 4 categories of <span style="color:red">Budget/Economy/Premium/Luxury</span>. This classification was done based on price point.
3. Streamlit framework has been used for front end web application and for visualization: Plotly and Seaborn mostly
4. This project uses GitHub Actions with a ***self-hosted*** runner to automate Docker image builds and deployment to the ML2023 server. Upon pushing changes to the main branch, the pipeline:
	- Builds and Pushes the Docker image to Docker Hub.
	- Connects to ML2023 via SSH using a self-hosted runner.
	- Pulls the latest image and restarts the container using docker compose.
 
5. I initially faced issues setting up SSH authentication in the GitHub Actions environment due to incorrect handling of the private key and SSH configuration. Despite adding the private key as    a GitHub secret, the SSH connection failed, likely due to formatting errors or permission issues. To resolve this, we switched to using a self-hosted runner, allowing our local machine to       handle SSH authentication directly, ensuring a smoother and more secure deployment process.
