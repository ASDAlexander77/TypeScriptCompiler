import fs from 'fs-extra';

import _ from 'underscore';

import { logger } from '../logger';
import * as utils from '../utils';

import { BaseCompiler } from '../base-compiler';

import { TypeScriptNativeParser } from './argument-parsers';

export class TypeScriptCompiler extends BaseCompiler {
    static get key() {
        return 'typescript';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.compiler.supportsIntel = false;
        this.compiler.supportsIrView = true;

        this.tscJit = this.compilerProps(`compiler.${this.compiler.id}.exe`);
        this.tscSharedLib = this.compilerProps(`compiler.${this.compiler.id}.sharedlibs`);
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    optionsForFilter(filters, outputFilename) {
        let options = [this.filename(outputFilename)];
        return options;
    }    

    async handleInterpreting(key, executeParameters) {
        executeParameters.args = [
            '--emit=jit',
            this.tscSharedLib ? '--shared-libs=' + this.tscSharedLib : '-nogc',
            ...executeParameters.args,
        ];

        return await super.handleInterpreting(key, executeParameters);
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        logger.debug("compile");

        // These options make Clang produce an IR
        const newOptions = [
            '--emit=mlir-llvm',
            inputFilename
        ];

        if (!this.tscSharedLib)
        {
            newOptions.push('-nogc');
        }

        const output = await this.runCompilerRawOutput(this.tscJit, newOptions, this.filename(inputFilename), execOptions);
        if (output.code !== 0) {
            return [{ text: 'Failed to run compiler to get MLIR code' }];
        }

        return { code: 0 };
    }

    async generateIR(inputFilename, options, filters) {

        logger.debug("Generate IR:");

        // These options make Clang produce an IR
        const newOptions = [
            '--emit=llvm',
            inputFilename
        ];

        if (!this.tscSharedLib)
        {
            newOptions.push('-nogc');
        }

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const output = await this.runCompilerRawOutput(this.tscJit, newOptions, this.filename(inputFilename), execOptions);
        if (output.code !== 0) {
            return [{ text: 'Failed to run compiler to get IR code' }];
        }

        filters.commentOnly = false;        
        filters.libraryCode = true;
        filters.directives = true;

        const ir = await this.llvmIr.process(output.stderr, filters);
        return ir.asm;
    }

    isCfgCompiler() {
        return true;
    }

    getArgumentParser() {
        return TypeScriptNativeParser;
    }
}
