**Assignment Detail** 
1. Kindly login to http://st124783.ml.brain.cs.ait.ac.th to view the web application and model in action.
2. I have changed the target variable and classified into 4 categories of <span style="color:red">Budget/Economy/Premium/Luxury</span>. This classification was done based on price point.
3. Streamlit framework has been used for front end web application and for visualization: Plotly and Seaborn mostly
4. The CI/CD pipeline involve:
   - Run the test which involves checking the model being registered to the mlflow
   - Run the test which involves checking the model saved locally input and its param
   - Run the test which involves checking the model output
   - Login in to Docker and Build the docker image 
   - SSH to the ml2023 server using private key and proxy jump
   - Go to st124783 directory and then pull the image and make it run
   - Finally the website is up and running
   
 

