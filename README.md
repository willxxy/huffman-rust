# Huffman Coding in Rust with Python Bindings

Just wanted to learn some different compression techniques and practice Rust.
I referenced [this repository](https://github.com/nayuki/Reference-Huffman-coding/tree/master) to translate to the Rust version mostly.

## Usage

1. `pip install maturin`

2. `cd rust_huff`

3. `maturin develop --release`

4. `../`

5. `python main.py`

## Example output

```
(ecg) william@hanjongcbookpro huffman-coding % python main.py
Original text: your text here
[00:00:00] ########################################       4/4       Compression completedTotal compression time: 217.125µs

Encoded data: [175, 184, 209, 201, 233, 12]

Huffman codes:
' ': [False, True, True]
'e': [False, False]
'h': [True, True, False, True]
'o': [True, True, True, True]
'r': [True, False, False]
't': [False, True, False]
'u': [True, False, True, True]
'x': [True, True, True, False]
'y': [True, False, True, False]
EOF: [True, True, False, False]

Character frequencies:
' ': 2
'e': 3
'h': 1
'o': 1
'r': 2
't': 2
'u': 1
'x': 1
'y': 1
EOF: 1

Decoded text: your text here
Decoding correct: True
```
