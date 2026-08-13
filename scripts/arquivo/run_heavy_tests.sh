#!/bin/bash
set -e

echo "=== Compilando Ferramenta de CLI TLV/DNA V4 ==="
cargo build --release --bin generic_dna

GENERIC_DNA="../target/release/generic_dna"

echo ""
echo "=== Preparando Bateria de Testes Pesados ==="
mkdir -p test_battery
cd test_battery

echo "1. Gerando Arquivo de Entropia Pura (10 MB de ruído aleatório)..."
dd if=/dev/urandom of=entropy.bin bs=1M count=10 status=none

echo "2. Copiando Executável Compilado Linux (Binário ELF complexo)..."
cp ../target/release/generic_dna ./binary.elf

echo "3. Criando Tarball de todo o código-fonte (Texto comprímido em gzip)..."
tar -czf source_code.tar.gz -C .. src iara_genesis

echo "4. Copiando o CHAMPION.dna atual (Metadados mesclados v3)..."
cp ../CHAMPION.dna ./champion_copy.dna

echo ""
echo "=== Testando Encoding para Formato DNA V4 (IARA GNSS) ==="
for file in entropy.bin binary.elf source_code.tar.gz champion_copy.dna; do
    echo " -> Empacotando $file ..."
    $GENERIC_DNA encode "$file" "${file}.dna"
done

echo ""
echo "=== Testando Decoding V4 para Arquivo Original ==="
for file in entropy.bin binary.elf source_code.tar.gz champion_copy.dna; do
    echo " -> Desempacotando ${file}.dna ..."
    $GENERIC_DNA decode "${file}.dna" "${file}.restored"
done

echo ""
echo "=== VALIDANDO INTEGRIDADE MATEMÁTICA (SHA256) ==="
ALL_PASSED=true

for file in entropy.bin binary.elf source_code.tar.gz champion_copy.dna; do
    HASH_ORIGINAL=$(sha256sum "$file" | awk '{print $1}')
    HASH_RESTORED=$(sha256sum "${file}.restored" | awk '{print $1}')
    
    if [ "$HASH_ORIGINAL" == "$HASH_RESTORED" ]; then
        echo "✅ PASSOU: $file (SHA256: ${HASH_ORIGINAL:0:16}... válido bilhão por bilhão de bits)"
    else
        echo "❌ FALHOU: $file (Original: $HASH_ORIGINAL != Restaurado: $HASH_RESTORED)"
        ALL_PASSED=false
    fi
done

echo ""
if [ "$ALL_PASSED" = true ]; then
    echo "🎉 BATERIA DE TESTES GNSS/DNA CLASSIFICADA COMO SUCESSO ABSOLUTO!"
    echo "A estrutura TLV V4 suporta qualquer formato binário, não perdendo nem meio bit (0% corrupção)."
else
    echo "⚠ ALERTA: HOUVE CORRUPÇÃO DE DADOS DURANTE O ROUNDTRIP."
fi
