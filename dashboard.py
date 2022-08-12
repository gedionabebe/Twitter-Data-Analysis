import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import plotly.express as px
import re


st.set_page_config(page_title="Twitter Data Dashboard", layout="wide")
st.header('Twitter Data Dashboard')

def loadData():

    df = pd.read_csv('data\processed_tweet_data.csv')
   
    return df



def selectLocAndAuth():
    st.title("General Twitter Data")
    st.markdown("This table show the total data collected from twitter")
    df = loadData()
    location = st.multiselect("choose Location of tweets", list(df['place'].unique()))
    sentiment = st.multiselect("choose Sentiment of tweets", list(df['sentiment'].unique()))

    if location and not sentiment:
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    elif sentiment and not location:
        df = df[np.isin(df, sentiment).any(axis=1)]
        st.write(df)
    elif sentiment and location:
        location.extend(sentiment)
        df = df[np.isin(df, location).any(axis=1)]
        st.write(df)
    else:
        st.write(df)

def barChart(data, title, X, Y):
    title = title.title()
    st.title(f'{title} Chart')
    msgChart = (alt.Chart(data).mark_bar().encode(alt.X(f"{X}:N", sort=alt.EncodingSortField(field=f"{Y}", op="values",
                order='ascending')), y=f"{Y}:Q"))
    st.altair_chart(msgChart, use_container_width=True)

def wordCloud():
    df = loadData()
    cleanText = ''
    for text in df['original_text']:
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
        text = re.sub(r'https?:\/\/\S+', '', text)
        text = re.sub(r'RT[\S]+', '', text)
        tokens = str(text).lower().split()

        cleanText += " ".join(tokens) + " "

    wc = WordCloud(width=650, height=450, background_color='white', min_font_size=5).generate(cleanText)
    st.title("Twitter Word Cloud")
    st.markdown("Word cloud of the most used words in tweets ")
    st.image(wc.to_array())

def stBarChart():
    df = loadData()
    dfCount = pd.DataFrame({'Tweet_count': df.groupby(['place'])['original_text'].count()}).reset_index()
    dfCount["original_author"] = dfCount["place"].astype(str)
    dfCount = dfCount.sort_values("Tweet_count", ascending=False)
    num = st.slider("Select number of Rankings", 0, 50, 5)
    title = f"Top {num} Locations Ranking By Number of tweets"
    barChart(dfCount.head(num), title, "original_author", "Tweet_count")


def langPie():
    df = loadData()
    dfLangCount = pd.DataFrame({'Tweet_count': df.groupby(['sentiment'])['original_text'].count()}).reset_index()
    dfLangCount["language"] = dfLangCount["sentiment"].astype(str)
    dfLangCount = dfLangCount.sort_values("Tweet_count", ascending=False)
    dfLangCount.loc[dfLangCount['Tweet_count'] < 10, 'lang'] = 'Other languages'
    st.title(" Sentiment Distribution")
    st.markdown("This is distribution of sentiment across the entire dataset")
    fig = px.pie(dfLangCount, values='Tweet_count', names='sentiment', width=500, height=350)
    fig.update_traces(textposition='inside', textinfo='percent+label')

    colB1, colB2 = st.columns([2.5, 1])

    with colB1:
        st.plotly_chart(fig)
    with colB2:
        st.write(dfLangCount)







page_names_to_funcs = {
    
    "Word Cloud": wordCloud,
    "Locations Ranking By Number of Tweets": stBarChart,
    "Sentiment Distribution Piechart": langPie,
    "Tweeter Data Table": selectLocAndAuth,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


