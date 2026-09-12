//
//  GPT2Tokenizer.swift
//  iosApp
//
//  Created by Juan Silva on 25/11/2024.
//  Copyright © 2024 orgName. All rights reserved.
//

import Foundation

class GPT2Tokenizer {
    // Special tokens
    private let startToken: Int32 = 50256  // <|endoftext|>
    private let padToken: Int32 = 50256    // Using endoftext as pad
    
    // Vocabulary and merges (BPE pairs)
    private var encoder: [String: Int32] = [:]
    private var decoder: [Int32: String] = [:]
    private var bpeRanks: [Pair: Int] = [:]
    
    struct Pair: Hashable {
        let first: String
        let second: String
    }
    
    init() {
        loadVocabulary()
        loadMerges()
    }
    
    private func loadVocabulary() {
        // Load vocabulary from JSON file
        guard let vocabUrl = Bundle.main.url(forResource: "encoder", withExtension: "json"),
              let vocabData = try? Data(contentsOf: vocabUrl),
              let vocab = try? JSONSerialization.jsonObject(with: vocabData) as? [String: Int] else {
            print("Failed to load vocabulary")
            return
        }
        
        // Convert to our format
        encoder = vocab.mapValues { Int32($0) }
        decoder = Dictionary(uniqueKeysWithValues: encoder.map { ($1, $0) })
    }
    
    private func loadMerges() {
        // Load BPE merges from file
        guard let mergesUrl = Bundle.main.url(forResource: "vocab", withExtension: "bpe"),
              let mergesString = try? String(contentsOf: mergesUrl) else {
            print("Failed to load merges")
            return
        }
        
        // Parse merges
        let lines = mergesString.components(separatedBy: .newlines)
        for (rank, line) in lines.enumerated() {
            let parts = line.split(separator: " ")
            if parts.count == 2 {
                let pair = Pair(first: String(parts[0]), second: String(parts[1]))
                bpeRanks[pair] = rank
            }
        }
    }
    
    private func bytesToUnicode() -> [UInt8: String] {
        // Initialize the byte-to-unicode mapping
        var bytes: [UInt8] = []
        bytes.append(contentsOf: UInt8(33)...UInt8(126))
        bytes.append(contentsOf: UInt8(161)...UInt8(172))
        bytes.append(contentsOf: UInt8(174)...UInt8(255))
        
        var chars = bytes.map { String(UnicodeScalar($0)) }
        var n = 0
        for b in 0...255 {
            if !bytes.contains(UInt8(b)) {
                bytes.append(UInt8(b))
                chars.append(String(UnicodeScalar(UInt8(n + 256))))
                n += 1
            }
        }
        
        return Dictionary(uniqueKeysWithValues: zip(bytes, chars))
    }
    
    private func getPairs(_ word: [String]) -> Set<Pair> {
        var pairs = Set<Pair>()
        for i in 0..<word.count-1 {
            pairs.insert(Pair(first: word[i], second: word[i + 1]))
        }
        return pairs
    }
    
    private func bpe(_ token: String) -> String {
        var word = token.map { String($0) }
        
        if word.isEmpty { return token }
        
        while true {
            let pairs = getPairs(word)
            if pairs.isEmpty { break }
            
            var minRank: (Pair, Int)? = nil
            for pair in pairs {
                if let rank = bpeRanks[pair] {
                    if minRank == nil || rank < minRank!.1 {
                        minRank = (pair, rank)
                    }
                }
            }
            
            if minRank == nil { break }
            
            let (pair, _) = minRank!
            var newWord: [String] = []
            var i = 0
            
            while i < word.count {
                if i < word.count - 1 && word[i] == pair.first && word[i + 1] == pair.second {
                    newWord.append(pair.first + pair.second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            
            word = newWord
        }
        
        return word.joined()
    }
    
    func encode(_ text: String) -> [Int32] {
        // Normalize string
        var normalized = text.replacingOccurrences(of: "'", with: "'")
        normalized = normalized.replacingOccurrences(of: """
        , with: "\"")
        normalized = normalized.replacingOccurrences(of:
 """, with: "\"")
        
        // ByteLevel encoding
        let byteToUnicode = bytesToUnicode()
        let encoded = Array(normalized.utf8).map { byteToUnicode[$0] ?? "" }.joined()
        
        // Apply BPE
        var bpeTokens: [Int32] = []
        let words = encoded.split(separator: " ")
        
        for word in words {
            let bpeResult = bpe(String(word))
            let tokens = bpeResult.split(separator: " ")
            bpeTokens.append(contentsOf: tokens.compactMap { encoder[String($0)] })
        }
        
        // Add start token
        bpeTokens.insert(startToken, at: 0)
        
        return bpeTokens
    }
    
    func decode(_ tokens: [Int32]) -> String {
        let bytes = tokens.compactMap { decoder[$0] }.joined()
        return String(bytes)
    }
}
