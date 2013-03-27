/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	
       				   --jasonp@boo.net 6/27/07

Modified:	Ben Buhrow
Date:		11/24/09
Purpose:	Port into Yafu-1.14.
--------------------------------------------------------------------*/

#include "qs.h"
#include "gmp_xface.h"

uint8 choose_multiplier_siqs(uint32 B, mpz_t n)
{
	uint32 i, j;
	uint32 num_primes = MIN(2 * B, NUM_TEST_PRIMES);
	double best_score;
	uint8 best_mult;
	double scores[NUM_MULTIPLIERS];
	uint32 num_multipliers;
	double log2n = zlog(n);

	/* measure the contribution of 2 as a factor of sieve
	   values. The multiplier itself must also be taken into
	   account in the score. scores[i] is the correction that
	   is implicitly applied to the size of sieve values for
	   multiplier i; a negative score makes sieve values
	   smaller, and so is better */

	for (i = 0; i < NUM_MULTIPLIERS; i++) {
		uint8 curr_mult = mult_list[i];
		uint8 knmod8 = (uint8)((curr_mult * mpz_get_ui(n)) % 8);
		double logmult = log((double)curr_mult);

		/* only consider multipliers k such than
		   k*n will not overflow an mp_t */

		if (log2n + logmult > (32 * MAX_DIGITS - 2) * LN2)
			break;

		scores[i] = 0.5 * logmult;
		switch (knmod8) {
		case 1:
			scores[i] -= 2 * LN2;
			break;
		case 5:
			scores[i] -= LN2;
			break;
		case 3:
		case 7:
			scores[i] -= 0.5 * LN2;
			break;
		/* even multipliers start with a handicap */
		}
	}
	num_multipliers = i;

	/* for the rest of the small factor base primes */

	for (i = 1; i < num_primes; i++) {
		uint32 prime = (uint32)spSOEprimes[i];
		double contrib = log((double)prime) / (prime - 1);
		uint32 modp = (uint32)mpz_tdiv_ui(n, prime);

		for (j = 0; j < num_multipliers; j++) {
			uint8 curr_mult = mult_list[j];
			//uint32 knmodp = mp_modmul_1(modp, curr_mult, prime);
			uint32 knmodp = (modp * curr_mult) % prime;

			/* if prime i is actually in the factor base
			   for k * n ... */

			if (knmodp == 0 || jacobi_1(knmodp, prime) == 1) {

				/* ...add its contribution. A prime p con-
				   tributes log(p) to 1 in p sieve values, plus
				   log(p) to 1 in p^2 sieve values, etc. The
				   average contribution of all multiples of p
				   to a random sieve value is thus

				   log(p) * (1/p + 1/p^2 + 1/p^3 + ...)
				   = (log(p) / p) * 1 / (1 - (1/p))
				   = log(p) / (p-1)

				   This contribution occurs once for each
				   square root used for sieving. There are two
				   roots for each factor base prime, unless
				   the prime divides k*n. In that case there
				   is only one root */

				if (knmodp == 0)
					scores[j] -= contrib;
				else
					scores[j] -= 2 * contrib;
			}
		}

	}

	/* use the multiplier that generates the best score */

	best_score = 1000.0;
	best_mult = 1;
	for (i = 0; i < num_multipliers; i++) {
		double score = scores[i];
		if (score < best_score) {
			best_score = score;
			best_mult = mult_list[i];
		}
	}
	return best_mult;
}

uint32 yafu_factor_list_add(fact_obj_t *obj, factor_list_t *list,
				mpz_t new_factor) {

	uint32 i, bitsleft;
	int isnew = 1;
	mpz_t tmpz;

	mpz_init(tmpz);

	//look to see if we've already included this one
	for (i=0; i<list->num_factors; i++)
	{
		mp_t2gmp(&list->final_factors[i]->factor, tmpz);
		isnew &= (mpz_cmp(tmpz,new_factor) != 0);
	}

	if (isnew)
	{
		list->final_factors[list->num_factors] = (final_factor_t *)malloc(
			sizeof(final_factor_t));
		gmp2mp_t(new_factor, &list->final_factors[list->num_factors]->factor);
		list->num_factors++;
	}

	//now determine if we are done based on the bits of factors found compared to
	//the bits in the original n
	bitsleft = obj->bits;
	for (i=0; i<list->num_factors; i++)
	{
		mp_t2gmp(&list->final_factors[i]->factor, tmpz);
		bitsleft -= mpz_sizeinbase(tmpz, 2);
	}

	mpz_clear(tmpz);
	return bitsleft;
}

/*--------------------------------------------------------------------*/
uint32 yafu_find_factors(fact_obj_t *obj, mpz_t n, 
		fb_element_siqs *factor_base, uint32 fb_size,
		qs_la_col_t *vectors, uint32 vsize, 
		siqs_r *relation_list,
		uint64 *null_vectors, uint32 multiplier,
		mpz_t *poly_a_list, poly_t *poly_list,
		factor_list_t *factor_list) {

	/* Perform the square root phase of MPQS. null_vectors
	   contains 64 linear dependencies of the relations in
	   vectors[], which has vsize elements. The code constructs
	   two numbers X and Y (mod n) that are each the product
	   of a huge number of factors derived from vectors[]. It
	   then computes gcd(X+-Y, n) for each linear dependency.
	   gcd values that are not 1 or n are a nontrivial factor
	   of n (or a product of nontrivial factors). For n the
	   product of two prime factors this will happen 2/3 of the 
	   time per dependency, on average.
	   
	   More specifically, vectors[] is a list of relations,
	   each of the form

   	   (a[i] * x[i] + b[i])^2 = prod(factors[i][]) mod n

	   X is a product of the (a[i] * x[i] + b[i]) and Y is
	   the product of sqrt(prod(factors[i][])), both mod n. 
	   The code never needs to calculate explicit square roots, 
	   because it knows the complete factorization of Y. Whether 
	   relation i contributes to dependency j is determined by 
	   the value of the j_th bit of null_vectors[i].

	   Because this implementation uses the double large prime
	   variation of MPQS, a single relation can actually be composed
	   of the product of several entities of the above form. In that
	   case, all entities are multiplied into X and Y (or all
	   are skipped).

	   Note that the code doesn't stop with one nontrivial
	   factor; it prints them all. If you go to so much work
	   and the other dependencies are there for free, why not
	   use them? */

	mpz_t factor, x, y, tmp, tmp2, sum, tmpn;
	uint32 i, j, k, m;
	uint64 mask;
	uint32 *fb_counts;
	uint32 large_primes[200], num_large_primes;
	uint32 num_relations, prime;
	siqs_r *relation;
	uint32 factor_found = 0;
	int bits;

	mpz_init(factor);
	mpz_init(x);
	mpz_init(y);
	mpz_init(tmp);
	mpz_init(tmp2);
	mpz_init(sum);
	mpz_init(tmpn);
	
	fb_counts = (uint32 *)malloc(fb_size * sizeof(uint32));
	mpz_set(tmpn, n);

	bits = 0;
	/* For each dependency */
	for (mask = 1; mask; mask <<= 1) {
		memset(fb_counts, 0, fb_size * sizeof(uint32));
		mpz_set_ui(x, 1);
		mpz_set_ui(y, 1);

		/* For each sieve relation */
		for (i = 0; i < vsize; i++) {

			/* If the relation is not scheduled to
			   contribute to x and y, skip it */

			if (!(null_vectors[i] & mask))
				continue;
			
			/* compute the number of sieve_values */

			num_large_primes = 0;
			num_relations = vectors[i].cycle.num_relations;

			/* for all sieve values */

			for (j = 0; j < num_relations; j++) {
				mpz_ptr a, b;
				poly_t *poly;
				uint32 sieve_offset;
				uint32 sign_of_index;

				relation = &relation_list[vectors[i].cycle.list[j]];
				
				/* reconstruct a[i], b[i], x[i] and
				   the sign of x[i]. Drop the subscript
				   from here on. */

				poly = poly_list + relation->poly_idx;
				b = poly->b;
				a = poly_a_list[poly->a_idx];
				//sieve_offset = relation->sieve_offset & 
				//				0x7fffffff;
				//sign_of_index = relation->sieve_offset >> 31;
				sieve_offset = relation->sieve_offset;
				sign_of_index = relation->parity;

				/* Form (a * sieve_offset + b). Note that 
				   sieve_offset can be negative; in that
				   case the minus sign is implicit. We don't
				   have to normalize mod n because there
				   are an even number of negative values
				   to multiply together */
	
				mpz_mul_ui(sum, a, sieve_offset); //zShortMul(a,sieve_offset,&sum);

				if (sign_of_index == POSITIVE)
					mpz_add(sum, sum, b); //zAdd(&sum,b,&tmp);
				else
					mpz_sub(sum, sum, b); //zSub(&sum,b,&tmp);
	
				/* multiply the sum into x */	
				mpz_mul(x, x, sum); //zModMul(&x,&sum,n,&x);
				mpz_tdiv_r(x, x, n); 
	
				/* do not multiply the factors associated 
				   with this relation into y; instead, just 
				   update the count for each factor base 
				   prime. Unlike ordinary MPQS, the list
				   of factors is for the complete factor-
				   ization of a*(a*x^2+b*x+c), so the 'a' 
				   in front need not be treated separately */
	
				for (k = 0; k < relation->num_factors; k++)
					fb_counts[relation->fb_offsets[k]]++;
					
				/* if the sieve value contains one or more
				   large primes, accumulate them in a 
				   dedicated table. Do not multiply them
				   into y until all of the sieve values
				   for this relation have been processed */

				for (k = 0; k < 2; k++) {
					prime = relation->large_prime[k];
					if (prime == 1)
						continue;

					for (m = 0; m < num_large_primes; m++) {
						if (prime == large_primes[2*m]){
							large_primes[2*m+1]++;
							break;
						}
					}
					if (m == num_large_primes) {
						large_primes[2*m] = prime;
						large_primes[2*m+1] = 1;
						num_large_primes++;
					}
				}
			}

			for (j = 0; j < num_large_primes; j++) {
				for (k = 0; k < large_primes[2*j+1]/2; k++) {
					mpz_set_ui(factor, large_primes[2 * j]); //factor.val[0] = large_primes[2*j];
					mpz_mul(y, y, factor); //zModMul(&y,&factor,n,&y);  
					mpz_tdiv_r(y, y, n);
				}
			}
		}

		/* For each factor base prime p, compute 
			p ^ ((number of times p occurs in y) / 2) mod n
		   then multiply it into y. This is enormously
		   more efficient than multiplying by one p at a time */

		for (i = MIN_FB_OFFSET; i < fb_size; i++) {
			uint32 mask2 = 0x80000000;
			uint32 exponent = fb_counts[i] / 2;
			uint32 prime = factor_base->prime[i];

			
			if (fb_counts[i] &0x1)
				printf("odd exponent found\n");
				

			if (exponent == 0)
				continue;

			mpz_set_ui(tmp, prime); 
			mpz_set_ui(factor, prime);

			while (!(exponent & mask2))
				mask2 >>= 1;
			for (mask2 >>= 1; mask2; mask2 >>= 1) {
				mpz_mul(tmp, tmp, tmp); //zModMul(&tmp,&tmp,n,&tmp2);
				mpz_tdiv_r(tmp, tmp, n);
				
				if (exponent & mask2) {
					mpz_mul(tmp, tmp, factor); //zModMul(&tmp,&factor,n,&tmp); 
					mpz_tdiv_r(tmp, tmp, n);
				}
			}
			mpz_mul(y, tmp, y); //zModMul(&tmp,&y,n,&y);  
			mpz_tdiv_r(y, y, n);
		}

		/* compute gcd(x+y, n). If it's not 1 or n, save it 
		   (and stop processing dependencies if the product 
		   of all the probable prime factors found so far equals 
		   n). See the comments in Pari's MPQS code for a proof 
		   that it isn't necessary to also check gcd(x-y, n) */

		mpz_add(tmp, x, y); //zAdd(&x,&y,&tmp);
		mpz_gcd(tmp, tmp, n); //zLEGCD(&tmp,n,&tmp2);
		if ((mpz_cmp(tmp, n) != 0) && (mpz_cmp_ui(tmp, 1) != 0)) {

			/* remove any factors of the multiplier 
			   before saving tmp, and don't save at all
			   if tmp contains *only* multiplier factors */
			if (multiplier > 1) {
				uint32 ignore_me = spGCD(multiplier,
						mpz_tdiv_ui(tmp, multiplier)); //zShortMod(&tmp, multiplier));
				if (ignore_me > 1) {
					mpz_tdiv_q_ui(tmp, tmp, ignore_me); //zShortDiv(&tmp, ignore_me, &tmp2);
					if (mpz_cmp_ui(tmp, 1) == 0)
						continue;
				}
			}

			//ignore composite factors for now...
			if (!mpz_probab_prime_p(tmp, NUM_WITNESSES))
				continue;

			//add the factor to our global list
			bits = yafu_factor_list_add(obj, factor_list, tmp);

			//check if only the multiplier remains
			if (abs(bits) < 8)
				break;

			//divide the factor out of our number
			mpz_tdiv_q(tmp2, tmpn, tmp); //zDiv(&tmpn, &tmp, &tmp2, &tmp3);

			//check if the remaining number is prime
			if (mpz_probab_prime_p(tmp2, NUM_WITNESSES))
			{
				//add it to our global factor list
				//printf("remaining cofactor is prime\n");
				bits = yafu_factor_list_add(obj, factor_list, tmp2);

				//then bail
				break;
			}

			//divide out the multiplier from the remaining number
			if (multiplier > 1) {
				uint32 ignore_me = spGCD(multiplier,
						mpz_tdiv_ui(tmp2, multiplier));
				if (ignore_me > 1) {
					mpz_tdiv_q_ui(tmp, tmp2, ignore_me);

					//check again if the remaining number is prime
					if (mpz_probab_prime_p(tmp, NUM_WITNESSES))
					{
						//add it to our global factor list
						//printf("remaining cofactor is prime after removing multiplier\n");
						bits = yafu_factor_list_add(obj, factor_list, tmp);

						//then bail
						break;
					}
				}
			}
		}
	}

	free(fb_counts);
	mpz_clear(factor);
	mpz_clear(x);
	mpz_clear(y);
	mpz_clear(tmp);
	mpz_clear(tmp2);
	mpz_clear(tmpn);
	mpz_clear(sum);
	return factor_found;
}

