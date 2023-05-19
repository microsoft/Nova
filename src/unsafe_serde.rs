#![allow(non_snake_case)]

use std::{
  io::Write,
  mem::size_of,
};

// This method currently enables undefined behavior, by exposing padding bytes.
#[inline]
pub unsafe fn typed_to_bytes<T>(slice: &[T]) -> &[u8] {
  std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * size_of::<T>())
}

/// this is **incredibly, INCREDIBLY** dangerous
#[inline]
pub unsafe fn entomb_T<T, W: Write>(_f: &T, _bytes: &mut W) -> std::io::Result<()> {
  Ok(())
}

/// this is **incredibly, INCREDIBLY** dangerous
#[inline]
pub unsafe fn exhume_T<'a, 'b, T>(_f: &mut T, bytes: &'a mut [u8]) -> Option<&'a mut [u8]> {
  Some(bytes)
}

#[inline]
pub fn extent_T<T>(_this: &T) -> usize {
  0
}

/// this is **incredibly, INCREDIBLY** dangerous
#[inline]
pub unsafe fn entomb_usize_usize_T<T, W: Write>(
  _f: &(usize, usize, T),
  _bytes: &mut W,
) -> std::io::Result<()> {
  Ok(())
}

/// this is **incredibly, INCREDIBLY** dangerous
#[inline]
pub unsafe fn exhume_usize_usize_T<'a, 'b, T>(
  _f: &mut (usize, usize, T),
  bytes: &'a mut [u8],
) -> Option<&'a mut [u8]> {
  Some(bytes)
}

#[inline]
pub fn extent_usize_usize_T<T>(_this: &(usize, usize, T)) -> usize {
  0
}

macro_rules! vec_abomonate {
  ($entomb_name:ident, $exhume_name:ident, $extent_name:ident, $entomb_inner_name:ident, $exhume_inner_name:ident, $extent_inner_name:ident, $inner_type:ty) => {
    #[inline]
    pub unsafe fn $entomb_name<T, W: Write>(
      this: &Vec<$inner_type>,
      write: &mut W,
    ) -> std::io::Result<()> {
      write.write_all(typed_to_bytes(&this[..]))?;
      for element in this.iter() {
        $entomb_inner_name(element, write)?;
      }
      Ok(())
    }

    #[inline]
    pub unsafe fn $exhume_name<'a, 'b, T>(
      this: &'a mut Vec<$inner_type>,
      bytes: &'b mut [u8],
    ) -> Option<&'b mut [u8]> {
      // extract memory from bytes to back our vector
      let binary_len = this.len() * size_of::<$inner_type>();
      if binary_len > bytes.len() {
        None
      } else {
        let (mine, mut rest) = bytes.split_at_mut(binary_len);
        let slice =
          std::slice::from_raw_parts_mut(mine.as_mut_ptr() as *mut $inner_type, this.len());
        std::ptr::write(
          this,
          Vec::from_raw_parts(slice.as_mut_ptr(), this.len(), this.len()),
        );
        for element in this.iter_mut() {
          let temp = rest; // temp variable explains lifetimes (mysterious!)
          rest = $exhume_inner_name(element, temp)?;
        }
        Some(rest)
      }
    }

    #[inline]
    pub fn $extent_name<T>(this: &Vec<$inner_type>) -> usize {
      let mut sum = size_of::<$inner_type>() * this.len();
      for element in this.iter() {
        sum += $extent_inner_name(element);
      }
      sum
    }
  };
}

vec_abomonate!(
  entomb_vec_usize_usize_T,
  exhume_vec_usize_usize_T,
  extent_vec_usize_usize_T,
  entomb_usize_usize_T,
  exhume_usize_usize_T,
  extent_usize_usize_T,
  (usize, usize, T)
);

vec_abomonate!(
  entomb_vec_T,
  exhume_vec_T,
  extent_vec_T,
  entomb_T,
  exhume_T,
  extent_T,
  T
);