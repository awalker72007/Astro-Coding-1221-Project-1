import os
import base64
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import litellm
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

custom_api_base = "https://litellmproxy.osu-ai.org/"
load_dotenv()
astro1221_key = os.getenv("ASTRO1221_API_KEY")

# Default number of clusters, will be changed by streamlit
num_clusters = 5
num_stars = 30

def generate_random_constellations():


    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    global num_stars
    global num_clusters
    if num_stars < num_clusters:
        return "Error: Number of clusters is more than the number of stars"
    else:
        x_vals = np.random.uniform(0, 10, num_stars)
        y_vals = np.random.uniform(0, 10, num_stars)
        sizes = np.random.randint(20, 100, num_stars)

    circle = Circle((0, 0), radius=5, color='black', fill = True)

    ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='datalim')

    ax.set_xlim(-4,4)
    ax.set_ylim(-3,3)

    x_vals = np.random.uniform(-4, 4, num_stars)
    y_vals = np.random.uniform(-3, 3, num_stars)

    coords = np.column_stack((x_vals, y_vals))
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(coords)
    labels = kmeans.labels_

    for cluster_id in range(kmeans.n_clusters):
        indices = np.where(labels == cluster_id)[0]
        n = len(indices)
        if n < 2:
            continue
        px = x_vals[indices]
        py = y_vals[indices]
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        for _ in range(n - 1):
            best_a, best_b, best_d2 = -1, -1, np.inf
            if n < 2:
                continue
            px = x_vals[indices]
            py = y_vals[indices]
            in_tree = np.zeros(n, dtype=bool)
            in_tree[0] = True
            for _ in range(n - 1):
                best_a, best_b, best_d2 = -1, -1, np.inf
                for a in np.where(in_tree)[0]:
                    for b in np.where(~in_tree)[0]:
                        d2 = (px[a] - px[b]) ** 2 + (py[a] - py[b]) ** 2
                        if d2 < best_d2:
                            best_d2, best_a, best_b = d2, a, b
                in_tree[best_b] = True
                idx_a, idx_b = indices[best_a], indices[best_b]
                connection = plt.Line2D(
                    [x_vals[idx_a], x_vals[idx_b]],
                    [y_vals[idx_a], y_vals[idx_b]],
                    color="white",
                    alpha=0.6,
                    linewidth=1,
                )
                ax.add_line(connection)

    ax.scatter(x_vals, y_vals, s=sizes, marker="*", c=kmeans.labels_, alpha=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.title("Random Constellations", fontsize=14, fontweight="bold", color="white")

    fig.savefig("astroplot.png")
    plt.close(fig)


def get_mythology_from_llm(image_path="astroplot.png"):
    """Call LLM with the constellation image to generate mythology."""
    if not astro1221_key:
        return None
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    response = litellm.completion(
        model="openai/GPT-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a Greek theologian, and have to create a story from the given stars. Use Greek myth stories as a base for these.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    }
                ],
            },
        ],
        max_tokens=1000,
        api_base=custom_api_base,
        api_key=astro1221_key,
        temperature=0.3,
    )
    if response and response.choices:
        return response.choices[0].message.content
    return None


def main():
    st.set_page_config(page_title="Constellations & Mythology", layout="wide")
    st.title("Random Constellations Generator with Mythology")
    st.write(
        "This is a constellation generator, where we incorporate LLM into generating mythology with the set of constellations you receive :)")

    global num_stars 
    num_stars = st.slider("num_stars", min_value = 1, max_value = 200)
    #Ability to change number of stars

    global num_clusters
    num_clusters = st.slider("num_clusters", min_value = 1, max_value = 20)
    #giving users the ability to change the amount of clusters

    if st.button("Generate new constellations"):
        with st.spinner("Creating constellations..."):
            generate_random_constellations()
        st.rerun()

    if os.path.exists("astroplot.png"):
        st.image("astroplot.png")

        if astro1221_key and st.button("Generate mythology stories"):
            with st.spinner("Asking the LLM for myths..."):
                story = get_mythology_from_llm()
            if story:
                st.subheader("Mythologies")
                st.write(story)
            else:
                st.warning("Could not get mythologies. Check your API key and connection.")
    else:
        st.info("Click **Generate new constellations** to create constellations.")


if __name__ == "__main__":
    main()
