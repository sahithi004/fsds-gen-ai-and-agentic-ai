import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import (
    word_tokenize, sent_tokenize, blankline_tokenize,
    WhitespaceTokenizer, wordpunct_tokenize
)
from nltk.util import ngrams
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

# -------------------------------
# Tokenization
# -------------------------------
def apply_tokenizer(text, choice):
    if choice == "Word Tokenize":
        return word_tokenize(text)
    elif choice == "Sentence Tokenize":
        return sent_tokenize(text)
    elif choice == "Blank Line Tokenize":
        return blankline_tokenize(text)
    elif choice == "Whitespace Tokenize":
        wt = WhitespaceTokenizer()
        return wt.tokenize(text)
    elif choice == "WordPunct Tokenize":
        return wordpunct_tokenize(text)
    else:
        return ["Invalid Choice"]

# -------------------------------
# WordCloud
# -------------------------------
def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=500,
        height=250,
        background_color='black',
        colormap='Accent',
        mode='RGBA'
    ).generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# -------------------------------
# Stemming
# -------------------------------
def apply_stemmers(words):
    pst = PorterStemmer()
    lst = LancasterStemmer()
    sbst = SnowballStemmer("english")

    results = []
    for w in words:
        results.append({
            "Word": w,
            "Porter": pst.stem(w),
            "Lancaster": lst.stem(w),
            "Snowball": sbst.stem(w)
        })
    return results

# -------------------------------
# Lemmatization
# -------------------------------
def apply_lemmatizer(words):
    lemmatizer = WordNetLemmatizer()
    results = []
    for w in words:
        results.append({"Word": w, "Lemma": lemmatizer.lemmatize(w)})
    return results

# -------------------------------
# Stopwords
# -------------------------------
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered = [w for w in words if w.lower() not in stop_words]
    return filtered

# -------------------------------
# Named Entity Recognition (NER)
# -------------------------------
def perform_ner(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)
    return chunked

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="NLP + NLG App", layout="wide")
st.title("📝 NLU + NLG Playground")

# Input text
st.subheader("Enter Your Text")
user_input = st.text_area("Paste your text here:", height=200)

# Tabs for features
tabs = st.tabs(["🔎 Tokenization", "📊 N-Grams", "🌱 Stemming", "🍂 Lemmatization", "🚫 Stopwords", "🧩 NER", "🌈 WordCloud"])

# --- Tokenization Tab ---
with tabs[0]:
    tokenizer_option = st.selectbox(
        "Select a tokenization method:",
        ["Word Tokenize", "Sentence Tokenize", "Blank Line Tokenize", "Whitespace Tokenize", "WordPunct Tokenize"]
    )
    if st.button("Run Tokenizer", key="tok"):
        if user_input.strip():
            result = apply_tokenizer(user_input, tokenizer_option)
            st.success(f"Tokenizer Applied: {tokenizer_option}")
            st.write(result)
        else:
            st.warning("⚠️ Please enter some text.")

# --- N-Grams Tab ---
with tabs[1]:
    n_value = st.slider("Select N for n-grams:", 2, 5, 2)
    if st.button("Generate N-Grams"):
        if user_input.strip():
            tokens = word_tokenize(user_input)
            n_grams = list(ngrams(tokens, n_value))
            st.success(f"{n_value}-grams generated:")
            st.write(n_grams)
        else:
            st.warning("⚠️ Please enter some text.")

# --- Stemming Tab ---
with tabs[2]:
    words_input = st.text_input(
        "Enter words (space separated):",
        "give given affected understanding play eaten affects maximum",
        key="stem_input"
    )
    if st.button("Run Stemming", key="stem_btn"):
        words = words_input.split()
        results = apply_stemmers(words)
        st.table(results)

# --- Lemmatization Tab ---
with tabs[3]:
    lem_input = st.text_input(
        "Enter words (space separated):",
        "give given affected understanding play eaten affects maximum",
        key="lemma_input"
    )
    if st.button("Run Lemmatization", key="lemma_btn"):
        words = lem_input.split()
        results = apply_lemmatizer(words)
        st.table(results)

# --- Stopwords Tab ---
with tabs[4]:
    if st.button("Remove Stopwords"):
        if user_input.strip():
            filtered = remove_stopwords(user_input)
            st.success("Stopwords removed:")
            st.write(filtered)
        else:
            st.warning("⚠️ Please enter some text.")

# --- NER Tab ---
with tabs[5]:
    if st.button("Run NER"):
        if user_input.strip():
            ner_tree = perform_ner(user_input)
            st.success("Named Entities:")
            st.text(ner_tree.pformat())  # pretty print tree
        else:
            st.warning("⚠️ Please enter some text.")

# --- WordCloud Tab ---
with tabs[6]:
    if st.button("Generate WordCloud"):
        if user_input.strip():
            generate_wordcloud(user_input)
        else:
            st.warning("⚠️ Please enter some text.")
