import tkinter as tk
from tkinter import PhotoImage
from tkinter import font
from main import calculate_embedding_glove, get_song_names, cosine_similarity

def find_closest_songs():
    input_text = entry.get()
    if input_text:
        input_embedding = calculate_embedding_glove(input_text)
        if input_embedding is not None:
            song_names = get_song_names(input_text)
            if song_names:
                song_similarities = []
                for song_info in song_names:
                    song_name = song_info['track']['track_name']
                    artist_name = song_info['track']['artist_name']
                    song_embedding = calculate_embedding_glove(song_name)
                    if song_embedding is not None:
                        similarity = cosine_similarity(input_embedding, song_embedding)
                        song_similarities.append((song_name, artist_name, similarity))
                song_similarities.sort(key=lambda x: x[2], reverse=True)
                song_similarities = song_similarities[:5]
                result_text.set("\n".join([f"{song_name} by {artist_name}" for song_name, artist_name, _ in song_similarities]))

root = tk.Tk()
root.title("LyricCat")
custom_font = font.Font(family='bahnschrift', size=12)
logo_image = PhotoImage(file='logo.png').subsample(2,2)
logo_label = tk.Label(root, image=logo_image)
logo_label.pack()
label = tk.Label(root, text="Enter a phrase or lyrics:", font=custom_font)
label.pack()
entry = tk.Entry(root, width=50)
entry.pack()
search_button = tk.Button(root, text="Search", command=find_closest_songs, font=custom_font)
search_button.pack()
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=custom_font)
result_label.pack()
root.mainloop()