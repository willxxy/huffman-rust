use pyo3::prelude::*;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use rayon::ThreadPoolBuilder;
use std::time::Instant;

#[derive(Debug, Eq, PartialEq)]
enum HuffmanNode {
    Leaf {
        symbol: u32,
        freq: u64,
    },
    Internal {
        freq: u64,
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

impl HuffmanNode {
    fn freq(&self) -> u64 {
        match self {
            HuffmanNode::Leaf { freq, .. } => *freq,
            HuffmanNode::Internal { freq, .. } => *freq,
        }
    }
}

#[derive(Eq)]
struct NodeWrapper(HuffmanNode);

impl Ord for NodeWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.freq().cmp(&self.0.freq())
    }
}

impl PartialOrd for NodeWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for NodeWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.0.freq() == other.0.freq()
    }
}

struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    position: u8,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            data: Vec::new(),
            current_byte: 0,
            position: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.position);
        }
        self.position += 1;

        if self.position == 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.position = 0;
        }
    }

    fn finish(&mut self) {
        if self.position > 0 {
            self.data.push(self.current_byte);
        }
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    current_byte: u8,
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            current_byte: if !data.is_empty() { data[0] } else { 0 },
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bit(&mut self) -> Option<bool> {
        if self.byte_pos >= self.data.len() {
            return None;
        }

        let bit = (self.current_byte & (1 << (7 - self.bit_pos))) != 0;
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.byte_pos += 1;
            if self.byte_pos < self.data.len() {
                self.current_byte = self.data[self.byte_pos];
            }
            self.bit_pos = 0;
        }

        Some(bit)
    }
}

fn build_huffman_tree(freqs: &HashMap<u32, u64>) -> HuffmanNode {
    let mut heap = BinaryHeap::new();

    for (&symbol, &freq) in freqs {
        heap.push(NodeWrapper(HuffmanNode::Leaf { symbol, freq }));
    }

    if heap.len() == 1 {
        let NodeWrapper(node) = heap.pop().unwrap();
        if let HuffmanNode::Leaf { symbol, freq } = node {
            heap.push(NodeWrapper(HuffmanNode::Leaf { symbol, freq }));
            heap.push(NodeWrapper(HuffmanNode::Leaf { symbol: symbol + 1, freq: 0 }));
        }
    }

    while heap.len() > 1 {
        let NodeWrapper(node1) = heap.pop().unwrap();
        let NodeWrapper(node2) = heap.pop().unwrap();

        let combined_freq = node1.freq() + node2.freq();
        heap.push(NodeWrapper(HuffmanNode::Internal {
            freq: combined_freq,
            left: Box::new(node1),
            right: Box::new(node2),
        }));
    }

    heap.pop().unwrap().0
}

fn build_code_map(node: &HuffmanNode, current_code: &mut Vec<bool>, code_map: &mut HashMap<u32, Vec<bool>>) {
    match node {
        HuffmanNode::Leaf { symbol, .. } => {
            code_map.insert(*symbol, current_code.clone());
        }
        HuffmanNode::Internal { left, right, .. } => {
            current_code.push(false);
            build_code_map(left, current_code, code_map);
            current_code.pop();

            current_code.push(true);
            build_code_map(right, current_code, code_map);
            current_code.pop();
        }
    }
}

#[pyfunction]
fn huffman_encode(
    text: &str,
    num_threads: usize,
) -> PyResult<(Vec<u8>, HashMap<u32, Vec<bool>>, HashMap<u32, u64>)> {
    let start_total = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    let pool = Arc::new(pool);

    let pb = ProgressBar::new(4);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("##-"));

    pb.set_message("Computing frequencies");
    let bytes = text.as_bytes();
    let mut freq_map = pool.install(|| {
        bytes.par_iter()
            .fold(
                || HashMap::new(),
                |mut acc, &byte| {
                    *acc.entry(byte as u32).or_insert(0) += 1;
                    acc
                }
            )
            .reduce(
                HashMap::new,
                |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                }
            )
    });
    
    freq_map.insert(256, 1);
    pb.inc(1);

    pb.set_message("Building Huffman tree");
    let tree = build_huffman_tree(&freq_map);
    pb.inc(1);

    pb.set_message("Building code map");
    let mut code_map = HashMap::new();
    build_code_map(&tree, &mut Vec::new(), &mut code_map);
    pb.inc(1);

    // Encode text
    pb.set_message("Encoding text");
    let mut writer = BitWriter::new();
    
    for &byte in bytes {
        let code = code_map.get(&(byte as u32))
            .expect("Symbol not found in Huffman code map");
        for &bit in code {
            writer.write_bit(bit);
        }
    }

    if let Some(eof_code) = code_map.get(&256) {
        for &bit in eof_code {
            writer.write_bit(bit);
        }
    }
    writer.finish();
    pb.inc(1);

    pb.finish_with_message("Compression completed");
    
    let total_duration = start_total.elapsed();
    println!("Total compression time: {:?}", total_duration);

    Ok((writer.data, code_map, freq_map))
}

#[pyfunction]
fn huffman_decode(
    encoded: Vec<u8>,
    code_map: HashMap<u32, Vec<bool>>,
    expected_length: usize,
) -> PyResult<String> {
    // Start with a single leaf. We'll expand it as needed.
    let mut decode_tree = HuffmanNode::Leaf { symbol: 0, freq: 0 };

    // Build the decode tree from the code map
    for (symbol, code) in code_map.iter() {
        let mut current = &mut decode_tree;

        for &bit in code.iter() {
            current = match current {
                HuffmanNode::Internal { left, right, .. } => {
                    if bit { &mut **right } else { &mut **left }
                }
                HuffmanNode::Leaf { .. } => {
                    // Need to go deeper, so convert this leaf into an internal node
                    *current = HuffmanNode::Internal {
                        freq: 0,
                        left: Box::new(HuffmanNode::Leaf { symbol: 0, freq: 0 }),
                        right: Box::new(HuffmanNode::Leaf { symbol: 0, freq: 0 }),
                    };
                    match current {
                        HuffmanNode::Internal { left, right, .. } => {
                            if bit { &mut **right } else { &mut **left }
                        }
                        _ => unreachable!(),
                    }
                }
            };
        }

        // Once all bits of the code have been processed, we are at the final node.
        // Place the symbol here.
        *current = HuffmanNode::Leaf {
            symbol: *symbol,
            freq: 0,
        };
    }

    // Now decode the data
    let mut reader = BitReader::new(&encoded);
    let mut result = Vec::with_capacity(expected_length);
    let mut current = &decode_tree;

    while result.len() < expected_length {
        let bit = match reader.read_bit() {
            Some(b) => b,
            None => break, // Out of bits before reaching EOF is strange, but just break
        };

        current = match current {
            HuffmanNode::Internal { left, right, .. } => {
                if bit { right } else { left }
            }
            HuffmanNode::Leaf { symbol, .. } => {
                if *symbol == 256 {
                    // EOF symbol
                    break;
                }
                result.push(*symbol as u8);
                &decode_tree
            }
        };

        // If we ended up at a leaf after reading a bit (i.e. if code boundary is here):
        if let HuffmanNode::Leaf { symbol, .. } = current {
            if *symbol == 256 {
                break;
            }
            result.push(*symbol as u8);
            current = &decode_tree;
        }
    }

    String::from_utf8(result)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UTF-8: {}", e)))
}


#[pymodule]
fn rust_huff(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(huffman_encode, m)?)?;
    m.add_function(wrap_pyfunction!(huffman_decode, m)?)?;
    Ok(())
}