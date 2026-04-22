#!/bin/bash
set -e

echo "=== Compilando Ferramenta de CLI TLV/DNA V4 ==="
cargo build --release --bin generic_dna

GENERIC_DNA="../target/release/generic_dna"

echo ""
echo "=== Preparando Bateria de Testes Extremos ==="
mkdir -p test_extreme
cd test_extreme

echo "1. Gerando 500 MB de dados aleatórios..."
dd if=/dev/urandom of=size_500MB.bin bs=1M count=500 status=none

echo "2. Gerando 2000 MB de dados aleatórios..."
dd if=/dev/urandom of=size_2000MB.bin bs=1M count=2000 status=none

echo "3. Gerando 5000 MB de preenchimento nulo (compressível)..."
# Usamos /dev/zero para um teste massivo que deve processar rapidamente na criação mas 
# validar o encoding completo via DNA V4. E testa a eficácia da leitura em memória.
dd if=/dev/zero of=size_5000MB_zeros.bin bs=1M count=5000 status=none

echo ""
echo "=== Testando Encoding para Formato DNA V4 (IARA GNSS) ==="
for file in size_500MB.bin size_2000MB.bin size_5000MB_zeros.bin; do
    echo " -> Empacotando $file (Isto pode consumir alguma memória RAM)..."
    time $GENERIC_DNA encode "$file" "${file}.dna"
done

echo ""
echo "=== Testando Decoding V4 para Arquivo Original ==="
for file in size_500MB.bin size_2000MB.bin size_5000MB_zeros.bin; do
    echo " -> Desempacotando ${file}.dna ..."
    time $GENERIC_DNA decode "${file}.dna" "${file}.restored"
done

echo ""
echo "=== VALIDANDO INTEGRIDADE MATEMÁTICA (SHA256) ==="
ALL_PASSED=true

for file in size_500MB.bin size_2000MB.bin size_5000MB_zeros.bin; do
    echo " -> Calculando Hash para $file ..."
    HASH_ORIGINAL=$(sha256sum "$file" | awk '{print $1}')
    HASH_RESTORED=$(sha256sum "${file}.restored" | awk '{print $1}')
    
    if [ "$HASH_ORIGINAL" == "$HASH_RESTORED" ]; then
        echo "✅ PASSOU: $file (SHA256 validado perfeitamente!)"
    else
        echo "❌ FALHOU: $file (Corrupção de bits)"
        ALL_PASSED=false
    fi
done

echo ""
if [ "$ALL_PASSED" = true ]; then
    echo "🔥 SUCESSO! O motor de serialização tolerou 500MB de peso por payload sem quebrar o limite de RAM ou corromper bits!"
else
    echo "⚠ ALERTA: HOUVE CORRUPÇÃO OU ESTOURO DE MEMÓRIA EM TESTES EXTREMOS."
fi
