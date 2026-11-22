use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input,parse_quote,ItemFn};
#[proc_macro_attribute]
pub fn kernel(_args:TokenStream,item:TokenStream)->TokenStream{
    let mut f=parse_macro_input!(item as ItemFn);
    assert!(f.sig.unsafety.is_some(),"device kernel must be unsafe");
    f.sig.abi=Some(parse_quote!(extern "ptx-kernel"));
    f.attrs.push(parse_quote!(#[unsafe(no_mangle)]));
    quote!(#f).into()
}
