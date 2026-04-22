# DNA V4 Genesis — Formato Universal de Compressão Generativa

> **"Não salve o oceano. Salve a fórmula da água."**

Este diretório documenta a arquitetura, os algoritmos e o plano de implementação do formato `.dna` V4 — um formato de armazenamento universal que **gera** dados em memória a partir de sementes matemáticas compactas, em vez de copiá-los byte a byte do disco.

## Índice

| # | Documento | Conteúdo |
|---|-----------|----------|
| 1 | [01_visao_geral.md](./01_visao_geral.md) | Filosofia, motivação, comparação com formatos existentes |
| 2 | [02_arquitetura_tlv.md](./02_arquitetura_tlv.md) | Estrutura binária TLV, layout do arquivo, tags definidas |
| 3 | [03_compressao_generativa.md](./03_compressao_generativa.md) | SVD, Seeds, Hypernetworks — como gerar tensores a partir de sementes |
| 4 | [04_streaming_mmap.md](./04_streaming_mmap.md) | I/O sem estourar RAM: mmap, BufWriter, zero-copy GPU |
| 5 | [05_quantizacao_tensores.md](./05_quantizacao_tensores.md) | Pipeline BF16→Q4→Ternary, integração com iara-engine |
| 6 | [06_fluxos_e_exemplos.md](./06_fluxos_e_exemplos.md) | Diagramas de fluxo, exemplos de código, casos de uso |
| 7 | [07_plano_implementacao.md](./07_plano_implementacao.md) | Fases, tarefas, dependências, critérios de validação |
| 8 | [08_representacao_implicita.md](./08_representacao_implicita.md) | **Pesos líquidos, INR universal, dados como equações** |

## Status Atual (2026-04-21)

- ✅ Formato TLV básico implementado (`iara_genesis/src/tlv.rs`)
- ✅ Codecs matemáticos: SparseRLE, Fourier, Chebyshev (`iara_genesis/src/codec/`)
- ✅ Pipeline DNA V4 integrado no simulador (`src/evolution/dna_genesis.rs`)
- ✅ Testes de integridade SHA256 até 500MB validados
- ⬜ Streaming I/O (mmap) para arquivos grandes
- ⬜ Quantização de tensores LLM (BF16→Q4)
- ⬜ Chunks geradores (TGEN) para compressão por fatoração
- ⬜ Integração com motor de inferência HIP/AMD (`llm-lab/iara-engine`)
- ⬜ Representação Implícita (SIREN/INR) para compressão universal
- ⬜ Pesos líquidos (lazy layer materialization)

## Projetos Relacionados

| Projeto | Path | Papel |
|---------|------|-------|
| Universe | `/home/leonardo/projects/LLM/Universe` | Simulação evolutiva, DNA V4, iara_genesis |
| llm-lab | `/home/leonardo/projects/LLM/llm-lab` | Motor de inferência Rust + HIP/AMD, formato IARA V2 |
| iara_genesis | `Universe/iara_genesis/` | Crate de compressão generativa (codec, TLV, neural) |
| iara-engine | `llm-lab/iara-engine/` | Engine de inferência com quantização Q4/FP8/Ternary |

## Modelo de Referência

```
Qwen 2.5 1.5B (test_extreme/qwen2.5-1.5b/)
├── model.safetensors    2.9 GB  (338 tensores, BF16)
├── quantized_cache.iara  938 MB  (Q4 group-128, gerado pelo llm-lab)
├── config.json          684 B   (28 layers, hidden=1536, vocab=151936)
├── tokenizer.json       6.8 MB
└── vocab.json           2.7 MB
```
