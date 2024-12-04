/// The `circuit2_witness` module implements witness-generation for the optimal Poseidon hash circuit.
use super::poseidon_inner::{Arity, Poseidon};
use crate::frontend::util_cs::witness_cs::SizedWitness;

use ff::PrimeField;
use generic_array::{sequence::GenericSequence, typenum::Unsigned, GenericArray};

impl<Scalar, A> SizedWitness<Scalar> for Poseidon<'_, Scalar, A>
where
  Scalar: PrimeField,
  A: Arity<Scalar>,
{
  fn num_constraints(&self) -> usize {
    let s_box_cost = 3;
    let width = A::ConstantsSize::to_usize();
    (width * s_box_cost * self.constants.full_rounds) + (s_box_cost * self.constants.partial_rounds)
  }

  fn num_inputs(&self) -> usize {
    0
  }

  fn num_aux(&self) -> usize {
    self.num_constraints()
  }
  fn generate_witness_into(&mut self, aux: &mut [Scalar], _inputs: &mut [Scalar]) -> Scalar {
    let width = A::ConstantsSize::to_usize();
    let constants = self.constants;
    let elements = &mut self.elements;

    let mut elements_buffer = GenericArray::<Scalar, A::ConstantsSize>::generate(|_| Scalar::ZERO);

    let c = &constants.compressed_round_constants;

    let mut offset = 0;
    let mut aux_index = 0;
    macro_rules! push_aux {
      ($val:expr) => {
        aux[aux_index] = $val;
        aux_index += 1;
      };
    }

    assert_eq!(width, elements.len());

    // First Round (Full)
    {
      // s-box
      for elt in elements.iter_mut() {
        let x = c[offset];
        let y = c[offset + width];
        let mut tmp = *elt;

        tmp.add_assign(x);
        tmp = tmp.square();
        push_aux!(tmp); // l2

        tmp = tmp.square();
        push_aux!(tmp); // l4

        tmp = tmp * (*elt + x) + y;
        push_aux!(tmp); // l5

        *elt = tmp;
        offset += 1;
      }
      offset += width; // post-round keys

      // mds
      {
        let m = &constants.mds_matrices.m;

        for j in 0..width {
          let scalar_product = m
            .iter()
            .enumerate()
            .fold(Scalar::ZERO, |acc, (n, row)| acc + (row[j] * elements[n]));

          elements_buffer[j] = scalar_product;
        }
        elements.copy_from_slice(&elements_buffer);
      }
    }

    // Remaining initial full rounds.
    {
      for i in 1..constants.half_full_rounds {
        // Use pre-sparse matrix on last initial full round.
        let m = if i == constants.half_full_rounds - 1 {
          &constants.pre_sparse_matrix
        } else {
          &constants.mds_matrices.m
        };
        {
          // s-box
          for elt in elements.iter_mut() {
            let y = c[offset];
            let mut tmp = *elt;

            tmp = tmp.square();
            push_aux!(tmp); // l2

            tmp = tmp.square();
            push_aux!(tmp); // l4

            tmp = tmp * *elt + y;
            push_aux!(tmp); // l5

            *elt = tmp;
            offset += 1;
          }
        }

        // mds
        {
          for j in 0..width {
            let scalar_product = m
              .iter()
              .enumerate()
              .fold(Scalar::ZERO, |acc, (n, row)| acc + (row[j] * elements[n]));

            elements_buffer[j] = scalar_product;
          }
          elements.copy_from_slice(&elements_buffer);
        }
      }
    }

    // Partial rounds
    {
      for i in 0..constants.partial_rounds {
        // s-box

        // FIXME: a little silly to use a loop here.
        for elt in elements[0..1].iter_mut() {
          let y = c[offset];
          let mut tmp = *elt;

          tmp = tmp.square();
          push_aux!(tmp); // l2

          tmp = tmp.square();
          push_aux!(tmp); // l4

          tmp = tmp * *elt + y;
          push_aux!(tmp); // l5

          *elt = tmp;
          offset += 1;
        }
        let m = &constants.sparse_matrixes[i];

        // sparse mds
        {
          elements_buffer[0] = elements
            .iter()
            .zip(&m.w_hat)
            .fold(Scalar::ZERO, |acc, (&x, &y)| acc + (x * y));

          for j in 1..width {
            elements_buffer[j] = elements[j] + elements[0] * m.v_rest[j - 1];
          }

          elements.copy_from_slice(&elements_buffer);
        }
      }
    }
    // Final full rounds.
    {
      let m = &constants.mds_matrices.m;
      for _ in 1..constants.half_full_rounds {
        {
          // s-box
          for elt in elements.iter_mut() {
            let y = c[offset];
            let mut tmp = *elt;

            tmp = tmp.square();
            push_aux!(tmp); // l2

            tmp = tmp.square();
            push_aux!(tmp); // l4

            tmp = tmp * *elt + y;
            push_aux!(tmp); // l5

            *elt = tmp;
            offset += 1;
          }
        }

        // mds
        {
          for j in 0..width {
            let scalar_product = m
              .iter()
              .enumerate()
              .fold(Scalar::ZERO, |acc, (n, row)| acc + (row[j] * elements[n]));

            elements_buffer[j] = scalar_product;
          }
          elements.copy_from_slice(&elements_buffer);
        }
      }

      // Terminal full round
      {
        // s-box
        for elt in elements.iter_mut() {
          let mut tmp = *elt;

          tmp = tmp.square();
          push_aux!(tmp); // l2

          tmp = tmp.square();
          push_aux!(tmp); // l4

          tmp *= *elt;
          push_aux!(tmp); // l5

          *elt = tmp;
        }

        // mds
        {
          for j in 0..width {
            elements_buffer[j] =
              (0..width).fold(Scalar::ZERO, |acc, i| acc + elements[i] * m[i][j]);
          }
          elements.copy_from_slice(&elements_buffer);
        }
      }
    }

    elements[1]
  }
}
