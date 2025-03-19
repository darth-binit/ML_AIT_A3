import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import os
from Model.Model import MyLogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

brand_means_path = os.path.join(BASE_DIR, "brand_means.pkl")
preprocess_path = os.path.join(BASE_DIR, "preprocess_test.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

brand_means = joblib.load(brand_means_path)
preprocessor = joblib.load(preprocess_path)
model = joblib.load(model_path)

data_file_path = os.path.join(BASE_DIR, "Data/Out_287.csv")
df = pd.read_csv(data_file_path)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.head(5))

## types of column

cat_col = df.select_dtypes(include='object').columns.tolist()
num_col = df.select_dtypes(include='number').columns.tolist()


##function for prediction
def predict(val):
    val['brand_encoded'] = val['brand'].map(brand_means)
    val['brand_encoded'] = val['brand_encoded'].fillna(np.mean(list(brand_means.values())))
    val = val.drop(columns=['brand'])
    preprocess = preprocessor.transform(val)
    _, prediction = model.predict(preprocess, is_test=True)

    return prediction

st.set_page_config(layout="wide")

def streamlit_menu():

    st.markdown(
        "<h1 style='text-align: center; color: #355F10;'>Chaky Automobiles Solution</h1>",
        unsafe_allow_html=True
    )

    selected = option_menu(
        menu_title=None,  # required
        options=["Home | Descriptive ", "Predictive Analytics"],  # required
        icons=["bi bi-activity", "bi bi-clipboard-data", "bi bi-pie-chart"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#C6DDB6", },
            "icon": {"color": "#072810", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",

                "padding":"10px"},
            "nav-link-selected": {"background-color": "#355F10"},
                    },)
    return selected

select  = streamlit_menu()

if select == "Home | Descriptive ":
    col1, col2, col3 = st.columns([1,0.13,1])
    col1_bg_color = "#c1d6c1"
    col2_bg_color = "#e4f2e4"

    with col2:
        icon_url = os.path.join(BASE_DIR, "Style", "auto-automobile-car-pictogram-service-traffic-transport--2.png")
        st.image(icon_url, width=50)

    with col1:
        st.markdown(
            f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA); font-size: 20px; padding: 8px; text-align: center;">Numerical Features Distribution</div>',
            unsafe_allow_html=True)
        feat_select = st.selectbox('Select Feature', df[num_col].columns)

        fig = px.histogram(
            df,
            x=feat_select,
            nbins=30,
            marginal="violin",  # Add KDE (marginal distribution)
            histnorm="density",
            opacity=0.75,
            color_discrete_sequence=['#379037'],
        )

        fig.update_layout(
            xaxis_title=feat_select,
            yaxis_title="Density",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)


    with col3:
        st.markdown(
            f'<div style = "background-image: linear-gradient(to left, #428142, #AAD4AA); font-size: 20px; padding: 8px; text-align: center;">Categorical Features Count</div>',
            unsafe_allow_html=True)
        cat_feat_select = st.selectbox('Select Feature', df[cat_col].columns)

        colors = ['#379037', '#71B971', '#9DD39D']
        feat_val = df[cat_feat_select].value_counts().values
        total = sum(feat_val)
        percentages = [f'{(v / total) * 100:.2f}%' for v in feat_val]

        trace = go.Bar(x=df[cat_feat_select].value_counts().index,
                       y=df[cat_feat_select].value_counts().values, marker=dict(color=colors),
                       hovertext=percentages)
        layout = go.Layout()
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('\n\n')
    st.markdown(
        f'<div style="background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;"> Correlation | Numerical Features </div>',
        unsafe_allow_html=True)
    st.markdown('\n\n\n\n')

    fig, ax = plt.subplots(figsize=(30, 16))
    num_corr = df[num_col].corr()
    mask = np.triu(np.ones_like(num_corr, dtype=bool))
    sns.heatmap(num_corr, mask=mask, xticklabels=num_corr.columns, yticklabels=num_corr.columns, annot=True, linewidths=.3,
                     cmap='Greens', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)


if select == 'Predictive Analytics':
    colp1, colp2 = st.columns([1,1])

    with colp2:
        prediction_placeholder = st.empty()
        st.session_state['prediction_result'] = ''
        prediction_placeholder.markdown(f"""<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>
                                                        <h2 style='text-align: center'>Prediction</h2>
                                                        <p style='font-size: 24px; text-align: center'>{st.session_state['prediction_result']}</p></div>""",
                                        unsafe_allow_html=True)

        st.markdown('\n\n')

        predict_btn = st.button('Predict', use_container_width=True)
        brand = st.selectbox('Brand', df['brand'].unique())

        colpp1, colpp2, colpp3 = st.columns([1,1,1])
        with colpp1:
            year = st.selectbox('Year', sorted(df['year'].unique()))
            transmission = st.selectbox('Transmission', df['transmission'].unique())
        with colpp2:
            seller_type = st.selectbox('Seller Type', df['seller_type'].unique())
            kms_driven = st.number_input('Kilometer Driven', min_value = 1000, max_value=2500000, value=1000)
        with colpp3:
            engine = st.number_input('Engine', min_value=500, max_value=4000, value=1400)
            max_power = st.number_input('Maximum Power', min_value=30, max_value=400, value = 85)

    usr_data = {'year': [year], 'km_driven': [kms_driven], 'seller_type': [seller_type],
                'transmission': [transmission], 'engine': [engine], 'max_power': [max_power],
                'brand': [brand]}
    usr_data = pd.DataFrame(usr_data)

    with colp1:
        sel_cols = ['year', 'km_driven', 'seller_type', 'transmission', 'engine', 'max_power',]
        seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
        transmission_map = {'Automatic': 2, 'Manual': 1}

        temp_df = df.drop(columns=['brand','selling_price','fuel','mileage','seats','owner'])
        temp_df = temp_df.replace(seller_map)
        temp_df = temp_df.replace(transmission_map)

        scaler = MinMaxScaler()
        scaled = scaler.fit(temp_df)

        mod_usr_data = usr_data.replace(seller_map)
        mod_usr_data = mod_usr_data.replace(transmission_map)
        mod_usr_data = mod_usr_data.drop(columns=['brand'])


        scaled_data = scaler.transform(mod_usr_data[sel_cols])

        scaled_df = pd.DataFrame(scaled_data, columns=sel_cols)

        fig = px.line_polar(scaled_df, scaled_df.values.reshape(-1), theta=sel_cols, line_close=True)
        fig.update_traces(fill='toself', line_color='green')
        st.plotly_chart(fig, use_container_width=True)


        st.markdown(f"""<div style='background-color:#c1d6c1 ; padding: 20px; border-radius: 3px'>
                                                        <h3 style='text-align: center'>Notes</h3>
                                                        <p style='font-size: 11px; text-align: center'> This is Classification machine learning model, where model has been trained based 
                                                          on used car data set that has been sold. Please fill all the value for field as given to the right. Incase of value not 
                                                            being available, we will set a default value based on our dataset distribution. Where ever you see feature name followed by
                                                              'v', it is drop down select menu, please select value given below and for the feature that hold '- +' that is where input needs to be
                                                              given by the user and then press enter. Finally when all the value is being entered, kindly press predict button to get classification of 
                                                              cars. We have classified car based on 0 for Budget, 1 for Economy, 2 for Premium, 3 for Luxury></div>""",
                                        unsafe_allow_html=True)

    if predict_btn:
        pred_val = predict(usr_data)[0]
        if pred_val == 0:
            pred_show = 'Budget 🚗'
        elif pred_val == 1:
            pred_show = 'Economy 🚕'
        elif pred_val == 2:
            pred_show = 'Premium 🚙'
        else:
            pred_show = 'Luxury 🚘'

        st.session_state['prediction_result'] = f"{pred_show}"
        print(st.session_state['prediction_result'])
        prediction_placeholder.markdown(f"""<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>
                                                        <h2 style='text-align: center'>Prediction | Category </h2>
                                                        <p style='font-size: 24px; text-align: center'>{st.session_state['prediction_result']}</p></div>""",
                                        unsafe_allow_html=True)



