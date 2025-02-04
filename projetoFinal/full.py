import os
import json
import re
from collections import defaultdict

import networkx as nx

# Load aliases from file
alias_dict = {}
with open("rappers.txt", 'r') as file:
    for line in file:
        parts = line.strip().split(';')
        primary_name = parts[0].strip()
        aliases = [alias.strip() for alias in parts[1:]]
        alias_dict[primary_name] = aliases

# Create a map from alias to primary name
alias_to_primary = {primary: primary for primary in alias_dict}
for primary, alias_list in alias_dict.items():
    alias_to_primary.update({alias: primary for alias in alias_list})

# Initialize the directed graph
G = nx.DiGraph()

# Process each JSON lyrics file
letras_folder = "./letras/"
for filename in os.listdir(letras_folder):
    if filename.endswith(".json"):
        # Map the artist name to its primary name
        artist_raw_name = filename.split("_")[1].replace('.json', '')
        artist_name = alias_to_primary.get(artist_raw_name, artist_raw_name)

        with open(os.path.join(letras_folder, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Extract lyrics
        lyrics_list = [song['lyrics'] for song in data['songs']]

        rapper_mentions = defaultdict(int)

        for lyric in lyrics_list:
            # Track mentions within a single song
            mentioned_in_song = set()

            # Then, proceed with the usual word-based check for single-word aliases
            words = re.findall(r'\b[A-Z][a-z]*\b', lyric)
            for word in words:
                if word in alias_to_primary:
                    primary_name = alias_to_primary[word]
                    if primary_name != artist_name:
                        # Increment the count of mentions
                        mentioned_in_song.add(primary_name)

            # Increment the mention count for each unique mention in the song
            for rapper in mentioned_in_song:
                rapper_mentions[rapper] += 1

        # Add nodes with labels and styling
        G.add_node(artist_name, label=artist_name)

        # Create edges with weights based on mention count across songs
        for rapper_primary_name, count in rapper_mentions.items():
            G.add_node(rapper_primary_name, label=rapper_primary_name)
            if G.has_edge(artist_name, rapper_primary_name):
                G[artist_name][rapper_primary_name]['weight'] += count
            else:
                G.add_edge(artist_name, rapper_primary_name, weight=count)

# Export the graph to a GML file
nx.write_gml(G, "rappers_graph.gml")

print("Graph has been exported to 'rappers_graph.gml'")
