/// Macros to give syntactic sugar for zipWith pattern and variants.
///
/// ```ignore
/// use crate::spartan::zip_with;
/// use itertools::Itertools as _; // we use zip_eq to zip!
/// let v = vec![0, 1, 2];
/// let w = vec![2, 3, 4];
/// let y = vec![4, 5, 6];
///
/// // Using the `zip_with!` macro to zip three iterators together and apply a closure
/// // that sums the elements of each iterator.
/// let res = zip_with!((v.iter(), w.iter(), y.iter()), |a, b, c| a + b + c)
///     .collect::<Vec<_>>();
///
/// println!("{:?}", res); // Output: [6, 9, 12]
/// ```

#[macro_export]
macro_rules! zip_with {
    // no iterator projection specified: the macro assumes the arguments *are* iterators
    // ```ignore
    // zip_with!((iter1, iter2, iter3), |a, b, c| a + b + c) ->
    //   iter1.zip_eq(iter2.zip_eq(iter3)).map(|(a, (b, c))| a + b + c)
    // ```
    //
    // iterator projection specified: use it on each argument
    // ```ignore
    // zip_with!(par_iter, (vec1, vec2, vec3), |a, b, c| a + b + c) ->
    //   vec1.par_iter().zip_eq(vec2.par_iter().zip_eq(vec3.par_iter())).map(|(a, (b, c))| a + b + c)
    // ````
    ($($f:ident,)? ($e:expr $(, $rest:expr)*), $($move:ident)? |$($i:ident),+ $(,)?| $($work:tt)*) => {{
        $crate::zip_with!($($f,)? ($e $(, $rest)*), map, $($move)? |$($i),+| $($work)*)
    }};
    // no iterator projection specified: the macro assumes the arguments *are* iterators
    // optional zipping function specified as well: use it instead of map
    // ```ignore
    // zip_with!((iter1, iter2, iter3), for_each, |a, b, c| a + b + c) ->
    //   iter1.zip_eq(iter2.zip_eq(iter3)).for_each(|(a, (b, c))| a + b + c)
    // ```
    //
    //
    // iterator projection specified: use it on each argument
    // optional zipping function specified as well: use it instead of map
    // ```ignore
    // zip_with!(par_iter, (vec1, vec2, vec3), for_each, |a, b, c| a + b + c) ->
    //   vec1.part_iter().zip_eq(vec2.par_iter().zip_eq(vec3.par_iter())).for_each(|(a, (b, c))| a + b + c)
    // ```
    ($($f:ident,)? ($e:expr $(, $rest:expr)*), $worker:ident, $($move:ident,)? |$($i:ident),+ $(,)?|  $($work:tt)*) => {{
        $crate::zip_all!($($f,)? ($e $(, $rest)*))
            .$worker($($move)? |$crate::nested_idents!($($i),+)| {
                $($work)*
            })
    }};
}

/// Like `zip_with` but use `for_each` instead of `map`.
#[macro_export]
macro_rules! zip_with_for_each {
    // no iterator projection specified: the macro assumes the arguments *are* iterators
    // ```ignore
    // zip_with_for_each!((iter1, iter2, iter3), |a, b, c| a + b + c) ->
    //   iter1.zip_eq(iter2.zip_eq(iter3)).for_each(|(a, (b, c))| a + b + c)
    // ```
    //
    // iterator projection specified: use it on each argument
    // ```ignore
    // zip_with_for_each!(par_iter, (vec1, vec2, vec3), |a, b, c| a + b + c) ->
    //   vec1.par_iter().zip_eq(vec2.par_iter().zip_eq(vec3.par_iter())).for_each(|(a, (b, c))| a + b + c)
    // ````
    ($($f:ident,)? ($e:expr $(, $rest:expr)*), $($move:ident)? |$($i:ident),+ $(,)?| $($work:tt)*) => {{
        $crate::zip_with!($($f,)? ($e $(, $rest)*), for_each, $($move)? |$($i),+| $($work)*)
    }};
}

// Foldright-like nesting for idents (a, b, c) -> (a, (b, c))
#[doc(hidden)]
#[macro_export]
macro_rules! nested_idents {
    ($a:ident, $b:ident) => {
        ($a, $b)
    };
    ($first:ident, $($rest:ident),+) => {
        ($first, $crate::nested_idents!($($rest),+))
    };
}

// Fold-right like zipping, with an optional function `f` to apply to each argument
#[doc(hidden)]
#[macro_export]
macro_rules! zip_all {
    (($e:expr,)) => {
        $e
    };
    ($f:ident, ($e:expr,)) => {
        $e.$f()
    };
    ($f:ident, ($first:expr, $second:expr $(, $rest:expr)*)) => {
        ($first.$f().zip_eq($crate::zip_all!($f, ($second, $( $rest),*))))
    };
    (($first:expr, $second:expr $(, $rest:expr)*)) => {
        ($first.zip_eq($crate::zip_all!(($second, $( $rest),*))))
    };
}
