import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dot, Activation, Concatenate
import pickle  # For saving and loading tokenizers
from sklearn.model_selection import train_test_split  # For dataset splitting

# -------------------------
# 🛠️ Part 1: Data Preprocessing & Splitting Dataset
# -------------------------

def read_txt(file_path):
    """ Reads text from a .txt file """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Read dataset
data = read_txt('dataset.txt')

# Updated regex pattern to match the specific dataset format
pattern = r'Code:\s*(.*?)\n#\s*(.*?)(?=\nCode:|$)'
matches = re.findall(pattern, data, re.DOTALL)

# Extract code snippets and comments from matches
code_snippets = []
comments = []

for code, comment in matches:
    code_snippets.append(code.strip())
    comments.append(comment.strip())

# Debugging: Print dataset stats
print("Total Code Snippets:", len(code_snippets))
print("Total Comments:", len(comments))
print("\nSample Extracted Code Snippets:", code_snippets[:2])
print("\nSample Extracted Comments:", comments[:2])

# Ensure dataset has enough samples before splitting
if len(code_snippets) == 0 or len(comments) == 0:
    raise ValueError("Dataset extraction failed! Check dataset format and regex patterns.")

# Split dataset (80% Training, 20% Testing)
train_x, test_x, train_y, test_y = train_test_split(
    code_snippets, comments, test_size=0.2, random_state=42
)

# Add start/end tokens to comments for sequence generation
train_y = ['start ' + comment + ' end' for comment in train_y]
test_y = ['start ' + comment + ' end' for comment in test_y]

# Tokenize code and comments
code_tokenizer = Tokenizer()
code_tokenizer.fit_on_texts(train_x)
code_sequences_train = code_tokenizer.texts_to_sequences(train_x)
code_sequences_test = code_tokenizer.texts_to_sequences(test_x)

comment_tokenizer = Tokenizer()
comment_tokenizer.fit_on_texts(train_y)
comment_sequences_train = comment_tokenizer.texts_to_sequences(train_y)
comment_sequences_test = comment_tokenizer.texts_to_sequences(test_y)

# Padding sequences
max_code_len = max(len(seq) for seq in code_sequences_train)
max_comment_len = max(len(seq) for seq in comment_sequences_train)

code_padded_train = pad_sequences(code_sequences_train, maxlen=max_code_len, padding='post')
code_padded_test = pad_sequences(code_sequences_test, maxlen=max_code_len, padding='post')

comment_padded_train = pad_sequences(comment_sequences_train, maxlen=max_comment_len, padding='post')
comment_padded_test = pad_sequences(comment_sequences_test, maxlen=max_comment_len, padding='post')

# Save tokenizers and sequence lengths for inference
with open("code_tokenizer.pkl", "wb") as f:
    pickle.dump(code_tokenizer, f)
with open("comment_tokenizer.pkl", "wb") as f:
    pickle.dump(comment_tokenizer, f)
# Save max lengths
with open("max_lengths.pkl", "wb") as f:
    pickle.dump((max_code_len, max_comment_len), f)

# -------------------------
# 🧠 Part 2: Build RNN Model with Attention
# -------------------------

# Define encoder (Code Input)
code_input = Input(shape=(max_code_len,))
code_embedding = Embedding(input_dim=len(code_tokenizer.word_index) + 1, output_dim=128)(code_input)

# Encoder LSTM
encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
encoder_output, state_h, state_c = encoder_lstm(code_embedding)

# Define decoder
decoder_input = Input(shape=(max_comment_len,))
decoder_embedding = Embedding(input_dim=len(comment_tokenizer.word_index) + 1, output_dim=128)(decoder_input)

decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Attention Layer
attention = Dot(axes=[2, 2])([decoder_output, encoder_output])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, encoder_output])

# Combine context with decoder output
decoder_combined = Concatenate(axis=-1)([context, decoder_output])

# Output layer
output_layer = Dense(len(comment_tokenizer.word_index) + 1, activation='softmax')(decoder_combined)

# Define model
model = Model([code_input, decoder_input], output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# 🚀 Part 3: Train the Model & Evaluate on Test Data
# -------------------------

# Prepare targets for training (shifted by one position for teacher forcing)
decoder_target_data = np.zeros_like(comment_padded_train)
for i in range(len(comment_padded_train)):
    decoder_target_data[i, :-1] = comment_padded_train[i, 1:]

# Train the model
model.fit(
    [code_padded_train, comment_padded_train],
    decoder_target_data,
    epochs=200,  # Reduced epochs for faster training, increase for better results
    batch_size=32,
    validation_split=0.1
)

# Save the model
model.save("code_comment_attention.keras")

# Evaluate model on test data
decoder_test_target_data = np.zeros_like(comment_padded_test)
for i in range(len(comment_padded_test)):
    decoder_test_target_data[i, :-1] = comment_padded_test[i, 1:]

test_loss, test_accuracy = model.evaluate([code_padded_test, comment_padded_test], decoder_test_target_data)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# -------------------------
# 🔥 Part 4: Inference Function for Generating Comments
# -------------------------

def generate_comment(code_text):
    """Generate a comment for the given code using beam search"""
    # Load saved artifacts
    try:
        with open("code_tokenizer.pkl", "rb") as f:
            code_tokenizer = pickle.load(f)
        with open("comment_tokenizer.pkl", "rb") as f:
            comment_tokenizer = pickle.load(f)
        with open("max_lengths.pkl", "rb") as f:
            max_code_len, max_comment_len = pickle.load(f)
        
        model = load_model("code_comment_attention.keras")
        
        # Process input code
        code_seq = code_tokenizer.texts_to_sequences([code_text])
        code_padded = pad_sequences(code_seq, maxlen=max_code_len, padding='post')
        
        # Initialize with start token
        state_value = None
        target_seq = np.zeros((1, max_comment_len))
        target_seq[0, 0] = comment_tokenizer.word_index.get('start', 1)
        
        # Output sequence
        output_words = []
        
        # Generate the output sequence word by word
        for i in range(max_comment_len - 1):
            output_tokens = model.predict([code_padded, target_seq], verbose=0)
            
            # Sample token with highest probability
            sampled_token_index = np.argmax(output_tokens[0, i, :])
            sampled_word = comment_tokenizer.index_word.get(sampled_token_index, '')
            
            # Exit condition: either hit max length or find stop word
            if sampled_word == 'end' or sampled_word == '':
                break
                
            if sampled_word != 'start':  # Don't add the start token to output
                output_words.append(sampled_word)
            
            # Update target sequence for next token prediction
            target_seq[0, i+1] = sampled_token_index
            
        return ' '.join(output_words)
    except Exception as e:
        return f"Error generating comment: {str(e)}"

# -------------------------
# 🎉 Part 5: Take User Input & Generate Comments
# -------------------------

while True:
    user_code = input("\nEnter your Python code snippet (or type 'exit' to quit):\n")
    if user_code.lower() == "exit":
        print("Exiting program. Goodbye!")
        break

    generated_comment = generate_comment(user_code)
    print("\nGenerated Comment:", generated_comment)