//
//  TorchModule.swift
//  iosApp
//
//  Created by Juan Silva on 23/11/2024.
//  Copyright © 2024 orgName. All rights reserved.
//
//
//import Foundation
//import LibTorch
//
//class TorchModule {
//    private var module: Module
//
//    init?(fileAtPath path: String) {
//        guard let module = Module(fileAtPath: path) else { return nil }
//        self.module = module
//    }
//
//    func forward(_ input: TorchTensor) -> TorchTensor? {
//        guard let output = module.forward([IValue.fromTensor(input)])?.toTensor() else {
//            return nil
//        }
//        return output
//    }
//}
