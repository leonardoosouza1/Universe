//! Testes de regressão da serialização do genoma — a barreira contra o Gap #7 do
//! CLAUDE.md do projeto ("campo novo no genoma sem atualizar o disco"). NÃO remover
//! (regra 9 do REFACTOR_PLAN.md §5).

use crate::genome_io::{save_genome, load_genome};
use universe::gpu::evolution::Genome;

/// Ida-e-volta do genoma: o que sai do disco tem que ser o que entrou.
///
/// # Por que este teste existe
///
/// O `save_genome` gravava L1, L2 e KAN. Quando o cérebro lento entrou no genoma
/// (1091 pesos) e depois o banco de τ (96), o serializador não foi atualizado — e o
/// campeão perdia esses campos em silêncio a cada `--carregar`, voltando com a
/// inicialização padrão, idêntica para toda a população. Gerações de seleção sobre a
/// modulação eram descartadas sem nenhum sintoma: nada falhava, nada avisava, os
/// números só ficavam piores do que deveriam.
///
/// Nenhuma leitura de código pega isso, porque cada função fazia exatamente o que
/// prometia. O que pega é comparar campo a campo os dois lados do disco. Este teste
/// falha automaticamente quando alguém adicionar o PRÓXIMO campo ao genoma e esquecer
/// o serializador — que é o erro real, e ele vai se repetir.
#[test]
fn genoma_sobrevive_ida_e_volta_no_disco() {
    let mut original = Genome::random(12345);
    original.randomize_tau(999);
    // Marca cada bloco com um valor impossível de sair da inicialização padrão, para
    // que "passou" não possa significar "os dois lados caíram no mesmo default".
    original.slow[0] = -1.234_5;
    let ultimo_slow = original.slow.len() - 1;
    original.slow[ultimo_slow] = 0.876_5;
    original.l2[7] = -0.999;

    let path = std::env::temp_dir().join("universe_genoma_ida_volta.bin");
    let path = path.to_str().unwrap();
    save_genome(&original, path).expect("gravou");
    let lido = load_genome(path).expect("leu");
    let _ = std::fs::remove_file(path);

    assert_eq!(lido.l1, original.l1, "L1 mudou no disco");
    assert_eq!(lido.l2, original.l2, "L2 mudou no disco");
    assert_eq!(lido.kan_knots, original.kan_knots, "KAN mudou no disco");
    assert_eq!(lido.slow, original.slow,
        "cérebro LENTO não sobreviveu ao disco — era exatamente este o bug");
    assert_eq!(lido.tau, original.tau,
        "banco de τ não sobreviveu ao disco");
}

/// Genoma gravado por uma versão antiga (sem lento nem τ) ainda carrega.
///
/// A compatibilidade aqui é por COMPRIMENTO, não por cabeçalho: `load_genome` só lê
/// os blocos finais se o arquivo for grande o bastante. Sem este teste, quem mexer no
/// formato quebra todos os campeões salvos e só descobre na próxima retomada.
#[test]
fn genoma_antigo_sem_lento_nem_tau_ainda_carrega() {
    use std::io::Write;
    let original = Genome::random(777);
    let path = std::env::temp_dir().join("universe_genoma_antigo.bin");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        for w in &original.l1 { f.write_all(&w.to_le_bytes()).unwrap(); }
        for w in &original.l2 { f.write_all(&w.to_le_bytes()).unwrap(); }
        for row in &original.kan_knots {
            for k in row { f.write_all(&k.to_le_bytes()).unwrap(); }
        }
    }
    let lido = load_genome(path.to_str().unwrap()).expect("arquivo antigo deve carregar");
    let _ = std::fs::remove_file(&path);

    assert_eq!(lido.l1, original.l1);
    // Sem o bloco no arquivo, cair no padrão é a verdade — aquele genoma realmente
    // não carregava o dado. O que não pode é ler lixo ou devolver vetor vazio.
    assert_eq!(lido.tau.len(), universe::gpu::evolution::N_CFC_CHANNELS);
    assert!(lido.tau.iter().all(|&d| d == universe::cfc::DECAY_DEFAULT));
    assert_eq!(lido.slow.len(), universe::engine::SlowBrain::new().all_weights().len());
}

/// Genoma v2 (com cabeçalho `UGEN`) sobrevive ida-e-volta no disco.
///
/// Complementa `genoma_sobrevive_ida_e_volta_no_disco`: aquele testa o formato
/// SEM cabeçalho (legado). Este garante que o novo formato versionado (magic +
/// versão) também é lido de volta corretamente pelo mesmo `load_genome`.
#[test]
fn genoma_v2_com_cabecalho_sobrevive_ida_e_volta() {
    let mut original = Genome::random(2026);
    original.randomize_tau(42);
    original.l1[0] = 0.123_4;
    original.slow[3] = -0.567_8;

    let path = std::env::temp_dir().join("universe_genoma_v2_cabecalho.bin");
    let path_str = path.to_str().unwrap();
    save_genome(&original, path_str).expect("gravou");

    // Confirma que o arquivo tem o magic "UGEN" + versão 2 no início — sem isso o
    // teste poderia passar mesmo se `save_genome` tivesse voltado a gravar sem
    // cabeçalho (load_genome tolera os dois formatos por sniff).
    let bytes = std::fs::read(path_str).expect("leu bytes crus");
    assert_eq!(&bytes[0..4], b"UGEN", "cabeçalho magic ausente no arquivo salvo");
    let versao = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(versao, 2, "versão do cabeçalho não é 2");

    let lido = load_genome(path_str).expect("leu");
    let _ = std::fs::remove_file(path_str);

    assert_eq!(lido.l1, original.l1, "L1 mudou no disco (v2)");
    assert_eq!(lido.l2, original.l2, "L2 mudou no disco (v2)");
    assert_eq!(lido.kan_knots, original.kan_knots, "KAN mudou no disco (v2)");
    assert_eq!(lido.slow, original.slow, "lento mudou no disco (v2)");
    assert_eq!(lido.tau, original.tau, "τ mudou no disco (v2)");
}

/// Blob legado SINTÉTICO (sem magic "UGEN", só os bytes crus de L1|L2|KAN) ainda
/// carrega pelo mesmo `load_genome` que lê o formato v2.
///
/// Diferente de `genoma_antigo_sem_lento_nem_tau_ainda_carrega` (que também testa
/// compatibilidade), este teste é o pedido explícito do E3: "loader aceita v2 E
/// legado sem magic (sniff primeiros 4 bytes)" — construído helpers puros, sem
/// depender de nenhum arquivo real gravado por versão antiga do binário.
#[test]
fn load_genome_detecta_legado_por_sniff_dos_primeiros_4_bytes() {
    let original = Genome::random(31337);
    let path = std::env::temp_dir().join("universe_genoma_legado_sniff.bin");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        for w in &original.l1 { f.write_all(&w.to_le_bytes()).unwrap(); }
        for w in &original.l2 { f.write_all(&w.to_le_bytes()).unwrap(); }
        for row in &original.kan_knots {
            for k in row { f.write_all(&k.to_le_bytes()).unwrap(); }
        }
    }
    // Confirma que os primeiros 4 bytes NÃO são o magic "UGEN" — o blob é
    // genuinamente legado, começando com bytes de peso f32 arbitrários.
    let bytes = std::fs::read(&path).unwrap();
    assert_ne!(&bytes[0..4], b"UGEN", "blob de teste deveria ser legado (sem magic)");

    let lido = load_genome(path.to_str().unwrap()).expect("blob legado sintético deve carregar");
    let _ = std::fs::remove_file(&path);

    assert_eq!(lido.l1, original.l1);
    assert_eq!(lido.l2, original.l2);
    assert_eq!(lido.kan_knots, original.kan_knots);
}

/// P0.5 (inicialização espectral de L1): genoma LEGADO carregado do disco NÃO é
/// re-escalado — herança intocada.
///
/// `Genome::random` (fonte da normalização espectral, `src/gpu/evolution.rs`) só roda
/// na CRIAÇÃO de um genoma novo. `load_genome` nunca chama `Genome::random` — monta o
/// `Genome` direto por struct literal a partir dos bytes lidos (genome_io.rs). Este
/// teste simula exatamente o caso real: um `.bin` salvo ANTES da hipótese P0.5 existir,
/// com L1 no ganho espectral cru que σ=0.30 i.i.d. sempre produziu (~7.0, ver
/// CLAUDE.md/docs/PLANO_EVOLUCAO.md §P0.5) — bem longe do alvo de 1.0. Se `load_genome`
/// algum dia passasse a "corrigir" o genoma ao carregar, este teste pega: o campeão
/// salvo por gerações de seleção mudaria de comportamento no meio da bateria, sem
/// aviso, ao ser retomado.
#[test]
fn genoma_legado_carregado_do_disco_nao_e_reescalado() {
    use universe::math::linalg::largest_singular_value;
    use universe::config::iara::brain::{L1_NEURONS, SENSORY_WAVE_SIZE};

    // Simula o L1 "legado": ganho espectral grande e conhecido, produzido sem
    // nenhuma normalização (era assim que TODO genoma nascia antes de P0.5).
    let mut legado = Genome::random(9999);
    for w in legado.l1.iter_mut() { *w *= 6.0; }
    let ganho_antes = largest_singular_value(&legado.l1, L1_NEURONS, SENSORY_WAVE_SIZE);
    assert!(
        ganho_antes > 2.0,
        "pré-condição do teste: ganho deveria estar bem longe de 1.0, obteve {ganho_antes}"
    );

    let path = std::env::temp_dir().join("universe_genoma_legado_ganho_p05.bin");
    let path_str = path.to_str().unwrap();
    save_genome(&legado, path_str).expect("gravou");
    let lido = load_genome(path_str).expect("leu");
    let _ = std::fs::remove_file(path_str);

    assert_eq!(lido.l1, legado.l1, "L1 mudou ao ir e voltar do disco — deveria ser byte-idêntico");
    let ganho_depois = largest_singular_value(&lido.l1, L1_NEURONS, SENSORY_WAVE_SIZE);
    assert!(
        (ganho_depois - ganho_antes).abs() < 1e-3,
        "ganho espectral mudou ao carregar: antes={ganho_antes}, depois={ganho_depois} — \
         genoma legado foi re-escalado, e não deveria ter sido"
    );
}
