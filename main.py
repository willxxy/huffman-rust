from rust_huff import huffman_encode, huffman_decode

test_text = "your text here"
print(f"Original text: {test_text}")

encoded_data, codes, frequencies = huffman_encode(test_text, num_threads=4)
print(f"\nEncoded data: {encoded_data}")
print(f"\nHuffman codes:")
for char, code in sorted(codes.items()):
    if char == 256:
        print(f"EOF: {code}")
    else:
        print(f"'{chr(char)}': {code}")

print(f"\nCharacter frequencies:")
for char, freq in sorted(frequencies.items()):
    if char == 256:
        print(f"EOF: {freq}")
    else:
        print(f"'{chr(char)}': {freq}")


decoded_text = huffman_decode(encoded_data, codes, len(test_text))
print(f"\nDecoded text: {decoded_text}")
print(f"Decoding correct: {decoded_text == test_text}")