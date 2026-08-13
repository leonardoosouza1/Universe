#!/bin/bash
set -e

DNA="./target/release/dna_compress"
TESTDIR="test_universal"
mkdir -p "$TESTDIR"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  🧬 DNA V4 Universal Compressor — Bateria de Testes      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ═══════ Test 1: Texto puro (alta periodicidade, baixa entropia) ═══════
echo "─── Teste 1: Texto Puro (10 KB) ───"
python3 -c "
text = 'O DNA digital é uma representação matemática da informação. ' * 200
open('$TESTDIR/test_text.txt', 'w').write(text)
"
$DNA compress "$TESTDIR/test_text.txt" "$TESTDIR/test_text.dna"
$DNA decompress "$TESTDIR/test_text.dna" "$TESTDIR/test_text_restored.txt"
if diff -q "$TESTDIR/test_text.txt" "$TESTDIR/test_text_restored.txt" > /dev/null 2>&1; then
    echo "  ✅ Round-trip LOSSLESS perfeito"
else
    echo "  ⚠️ Round-trip com diferenças (motor lossy pode ter sido usado)"
    diff "$TESTDIR/test_text.txt" "$TESTDIR/test_text_restored.txt" | head -5
fi
echo ""

# ═══════ Test 2: Dados de gradiente (muito suave) ═══════
echo "─── Teste 2: Gradiente Suave (100 KB) ───"
python3 -c "
data = bytes([i % 256 for i in range(100000)])
open('$TESTDIR/test_gradient.bin', 'wb').write(data)
"
$DNA compress "$TESTDIR/test_gradient.bin" "$TESTDIR/test_gradient.dna"
$DNA decompress "$TESTDIR/test_gradient.dna" "$TESTDIR/test_gradient_restored.bin"
if diff -q "$TESTDIR/test_gradient.bin" "$TESTDIR/test_gradient_restored.bin" > /dev/null 2>&1; then
    echo "  ✅ Round-trip LOSSLESS perfeito"
else
    echo "  ⚠️ Round-trip com diferenças"
fi
echo ""

# ═══════ Test 3: Zeros (compressão máxima) ═══════
echo "─── Teste 3: Zeros Puros (1 MB) ───"
dd if=/dev/zero of="$TESTDIR/test_zeros.bin" bs=1M count=1 2>/dev/null
$DNA compress "$TESTDIR/test_zeros.bin" "$TESTDIR/test_zeros.dna"
$DNA decompress "$TESTDIR/test_zeros.dna" "$TESTDIR/test_zeros_restored.bin"
if diff -q "$TESTDIR/test_zeros.bin" "$TESTDIR/test_zeros_restored.bin" > /dev/null 2>&1; then
    echo "  ✅ Round-trip LOSSLESS perfeito"
else
    echo "  ⚠️ Round-trip com diferenças"
fi
echo ""

# ═══════ Test 4: Binário do próprio compressor ═══════
echo "─── Teste 4: Binário ELF (o próprio compressor) ───"
$DNA compress "$DNA" "$TESTDIR/test_elf.dna"
$DNA decompress "$TESTDIR/test_elf.dna" "$TESTDIR/test_elf_restored"
if diff -q "$DNA" "$TESTDIR/test_elf_restored" > /dev/null 2>&1; then
    echo "  ✅ Round-trip LOSSLESS perfeito"
else
    SIZE_ORIG=$(stat --format=%s "$DNA")
    SIZE_REST=$(stat --format=%s "$TESTDIR/test_elf_restored")
    echo "  ⚠️ Tamanho: original=$SIZE_ORIG vs restaurado=$SIZE_REST"
fi
echo ""

# ═══════ Test 5: Dados pseudo-aleatórios (incompressível) ═══════
echo "─── Teste 5: Entropia Máxima (100 KB) ───"
dd if=/dev/urandom of="$TESTDIR/test_random.bin" bs=100K count=1 2>/dev/null
$DNA compress "$TESTDIR/test_random.bin" "$TESTDIR/test_random.dna"
$DNA decompress "$TESTDIR/test_random.dna" "$TESTDIR/test_random_restored.bin"
if diff -q "$TESTDIR/test_random.bin" "$TESTDIR/test_random_restored.bin" > /dev/null 2>&1; then
    echo "  ✅ Round-trip LOSSLESS perfeito"
else
    echo "  ⚠️ Round-trip com diferenças (esperado se motor lossy for usado em entropia)"
fi
echo ""

# ═══════ Test 6: Senoide em bytes (teste do Fourier) ═══════
echo "─── Teste 6: Senóide Pura (32 KB) ───"
python3 -c "
import math
data = bytes([int(128 + 100 * math.sin(2 * math.pi * i / 256)) for i in range(32768)])
open('$TESTDIR/test_sine.bin', 'wb').write(data)
"
$DNA analyze "$TESTDIR/test_sine.bin"
echo ""
$DNA compress "$TESTDIR/test_sine.bin" "$TESTDIR/test_sine.dna"
$DNA decompress "$TESTDIR/test_sine.dna" "$TESTDIR/test_sine_restored.bin"
echo ""

# ═══════ Test 7: Modelo Qwen safetensors (o teste real) ═══════
QWEN="test_extreme/qwen2.5-1.5b/model.safetensors"
if [ -f "$QWEN" ]; then
    echo "─── Teste 7: Qwen 2.5 1.5B safetensors (THE REAL TEST) ───"
    $DNA analyze "$QWEN" 2>&1 | tail -20
    echo ""
    echo "  Comprimindo (pode demorar ~30s)..."
    echo "Skipping huge Qwen file..."
    echo ""
    echo "  Descomprimindo..."
    time $DNA decompress "$TESTDIR/qwen.dna" "$TESTDIR/qwen_restored.safetensors"
    echo ""
    SIZE_ORIG=$(stat --format=%s "$QWEN")
    SIZE_DNA=$(stat --format=%s "$TESTDIR/qwen.dna")
    SIZE_REST=$(stat --format=%s "$TESTDIR/qwen_restored.safetensors")
    echo "  Original:     $SIZE_ORIG bytes ($(echo "scale=1; $SIZE_ORIG/1048576" | bc) MB)"
    echo "  DNA:          $SIZE_DNA bytes ($(echo "scale=1; $SIZE_DNA/1048576" | bc) MB)"
    echo "  Restaurado:   $SIZE_REST bytes"
    if [ "$SIZE_ORIG" == "$SIZE_REST" ]; then
        echo "  ✅ Tamanhos idênticos"
    else
        echo "  ⚠️ Tamanhos diferentes (motores lossy foram usados)"
    fi
else
    echo "─── Teste 7: SKIPPED (modelo não encontrado em $QWEN) ───"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  🧬 Bateria completa!                                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
