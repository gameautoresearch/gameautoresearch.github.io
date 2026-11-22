//! Derive literal indices from the exact embedded card table, never a hand map.
use std::{env,fs,path::PathBuf};
fn main(){
    let manifest=PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source=if manifest.join("data/cards.json").exists(){manifest.join("data/cards.json")}
        else{manifest.join("../../core/data/cards.json")};
    println!("cargo:rerun-if-changed={}",source.display());
    let text=fs::read_to_string(&source).expect("embedded card data");
    let data:serde_json::Value=serde_json::from_str(&text).expect("pinned card JSON");
    let cards=data["cards"].as_array().expect("card array");
    assert!(cards.len()<=4096);
    let mut rows=Vec::new();let mut fingerprint=0xcbf29ce484222325u64;
    for (i,card) in cards.iter().enumerate(){
        let name=card["id"].as_str().expect("card name");
        for b in name.bytes().chain(std::iter::once(0)){
            fingerprint=(fingerprint^b as u64).wrapping_mul(0x0100000001b3);
        }
        rows.push((name,i));
    }
    rows.sort_unstable_by_key(|row|row.0);
    assert!(rows.windows(2).all(|row|row[0].0!=row[1].0),"duplicate literal ID");
    let mut output=format!("pub const LITERAL_CARD_TABLE_FINGERPRINT:u64={fingerprint};\nconst LITERAL_CARD_IDS:&[(&str,u16)]=&[\n");
    for (name,index) in rows{output.push_str(&format!("({name:?},{index}),\n"));}
    output.push_str("];\n");
    fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("card_ids.rs"),output).unwrap();
}
