/* Author: Conor Stokes - Copyright 2017
 * Single header library for jeffreys divergence/centroids k-means of multiple weighted normalized histograms.
 */

#ifndef JEFC_H__
#define JEFC_H__

#ifdef JEFC_DOUBLE 
typedef double jefc_value_t;
#else
typedef float jefc_value_t;
#endif

#ifdef JEFC_STATIC_INLINE 
#define JEFC_IMPLEMENATION_TYPE inline static
#elif defined(JEFC_STATIC) 
#define JEFC_IMPLEMENTATION_TYPE static
#else
#define JEFC_IMPLEMENTATION_TYPE
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif 


/*
 * Weighted histogram for use in jeffreys k-means.
 */
typedef struct jefc_kmeans_histogram_t
{
    jefc_value_t* bins;   /* Note, for k-means should be normalized histogram with no zero buckets */
    jefc_value_t  weight; /* The weight that will be applied to this individual histogram among the histograms that make up the value. */
}
jefc_kmeans_histogram_t;

/*
 * A set of weighted histograms that counts as a single row value in jeffreys k-means.
 */
typedef struct jefc_kmeans_value_t
{
    jefc_kmeans_histogram_t* histograms; /* The histograms that make up this value. Note, the number of these
                                          * will be the same for all values used in a k-means. */
} 
jefc_kmeans_value_t;

/*
 * An output centroid from k-means
 * The actual centroid histograms are technically the jeffreys centroid of the values closest to this centroid in the previous iteration.
 * The means are the means of the values which are currently closest this centroid's histogram, which will usually be used
 * to build the next centroid step in a k-means iteration. 
 *
 * Currently the k-means implementation is monolithic, but having all this state makes it easy to split out the individual iterations.
 */
typedef struct jefc_kmeans_centroid_t
{
    jefc_kmeans_histogram_t* histograms; /* The actual centroid (made up of multiple histograms, mirroring the shape of jefc_kmeans_value_t. */
    jefc_value_t**           normalizedArithmeticMeans; /* The normalized arithmetic means, corresponding to the centroid histograms */
    jefc_value_t**           normalizedGeometricMeans;  /* The normalized geometric means, corresponding to the centroid histograms */
    size_t                   valueCount; /* The number of values for this. */
} 
jefc_kmeans_centroid_t;

/*
 * Centroid reference for a value
 * What centroid is a value closest to and what is the jeffreys divergence distance to it?
 */
typedef struct jefc_centroid_reference_t
{
    size_t       index;
    jefc_value_t distance;
}
jefc_centroid_reference_t;

/*
 * Error return codes.
 */
typedef enum jefc_error_t
{
    JEFC_SUCCESS = 0,
    JEFC_TOO_FEW_ITEMS = 1,
    JEFC_INVALID_CENTROID_COUNT = 2,
    JEFC_NULL_PARAMETER = 3,
    JEFC_ALLOC_FAILED = 4,
    JEFC_CONTEXT_TOO_SMALL = 5
}
jefc_error_t;

/*
 * Random number generator context.
 */
typedef struct jefc_random_t
{
    uint64_t state;
    uint64_t offset;
}
jefc_random_t;

/*
 * Seed a random number generator context.
 */
void jefc_seed( jefc_random_t* context, uint64_t stateSeed, uint64_t offsetSeed );

/*
 * Generate the next random number from a context.
 */
uint32_t jefc_random( jefc_random_t* context );

/*
 * Jeffreys divergence of two values (symmetric divergence metric of the relative cost of entropy 
 */
jefc_value_t jefc_jeffreys_divergence( const jefc_value_t* p, const jefc_value_t* q, size_t binCount );

/*
 * Extended version of the Kullback-Leibler divergence
 */
jefc_value_t jefc_extended_kl_divergence( const jefc_value_t* p, const jefc_value_t* q, size_t binCount );

/*
 * The Kullback-Leibler divergence (the extra data cost per datum of encoding a data set with 
 * probability distribution p using the (wrong) probability distribution q).
 * histogramSize is the number of bins in both p and q.
 */
jefc_value_t jefc_kl_divergence( const jefc_value_t* p, const jefc_value_t* q, size_t binCount );

/*
 * Calculate the positive Jeffreys centroid given an arithmetic weighted mean and geometric weighted mean of a set of histograms (with non-zero buckets). 
 * Doesn't return a normalized histogram when the input set of histograms are normalized (i.e. frequency histograms/discrete probability distributions).
 * For that purpose, use the jefc_jeffreys_frequency_centroid function.
 */
void jefc_jeffreys_positive_centroid( const jefc_value_t* arithmeticWeightedMean, 
                                      const jefc_value_t* geometricWeightedMean, 
                                      size_t binCount, 
                                      /* out */ jefc_value_t* centroid );

/*
 * Calculate the Jeffreys frequency centroid
 */
void jefc_jeffreys_frequency_centroid( const jefc_value_t* normalizedArithmeticWeightedMean, 
                                       const jefc_value_t* normalizedGeometricWeightedMean, 
                                       size_t binCount, 
                                       /* out */ jefc_value_t* centroid );

/*
 * Allocate k-means values and centroid references as one contiguous allocation, which can be free'd using the free
 * function corresponding to the function pointer passed by memoryAlloc, by freeing the pointer returned in values
 */
jefc_error_t jefc_allocate_kmeans_values(
    void* (*memoryAlloc)( size_t size ),
    size_t valueCount,
    size_t histogramCount,
    const size_t* binCounts,
    /* out */ jefc_kmeans_value_t** values,
    /* out */ jefc_centroid_reference_t** centroidReferences );

/*
 * Allocate a centroids list from a single allocation (pointer pointed to by centroids - just free this pointer to free the allocation).
 */
jefc_error_t jefc_allocate_kmeans_centroids(
    void* (*memoryAlloc)( size_t size ),
    size_t centroidCount,
    size_t histogramCount,
    const size_t* binCounts,
    /* out */ jefc_kmeans_centroid_t** centroids );

/*
 * Perform a (weighted) k-means using multiple non-zero normalized histograms/discrete probability distributions, using
 * (weighted) jeffreys divergence as a distance metric and (weighted) Jeffrey's centroids.
 * Params:
 *    values             - The items to perform k-means on, where each item consists of one or more normalized histograms (with non-zero bins), 
 *                         each with a probability weight as to its importance.
 *                         Each value has the same number of normalized histograms (histogramCount).
 *    valueCount         - The number of items in values. 
 *    histogramCount     - The number of histograms in each value.
 *    binCounts          - There are histogramCount histograms in each value, each histogram[ i ] has binCount[ i ] bins.
 *    maxIterations      - The maximum number of iterations to run k-means.
 *    centroidCount      - The number of output centroids.
 *    centroids          - The output centroids.
 *    centroidReferences - The references for each value to its centroid. Note, the output centroid is actually the centroid for the previous pass,
 *                         where as these references are the values clustered to that centroid.
 */
jefc_error_t jefc_jeffreys_weighted_kmeans(
    const jefc_kmeans_value_t* values,
    size_t valueCount,
    size_t histogramCount,
    const size_t* binCounts,
    size_t maxIterations,
    uint64_t randomSeed,
    size_t centroidCount,
    /*out - preallocated */ jefc_kmeans_centroid_t* centroids,
    /*out - preallocated */ jefc_centroid_reference_t* centroidReferences );

/*
 * Special case z >= 0 lambert W branch zero.
 */
jefc_value_t jefc_lambert_w0_positive( jefc_value_t z );

/*
 * Smooth a normalized histogram to get rid of zero buckets. 
 * This is useful for when we take geometric means, because a single "zero" 
 */
void jefc_smooth( const jefc_value_t* normalizedHistogram, 
                  size_t binCount, 
                  float smoothingValue,
                  /* out */ jefc_value_t* smoothedHistogram );

/*
 * Normalize a histogram in-place. i.e. make the sum of histogram values sum to 1
 */
void jefc_normalize_inplace( /* in out */ jefc_value_t* histogram, 
                             size_t binCount );

/*
 * Normalize a histogram. i.e. make the sum of histogram values sum to 1
 */
void jefc_normalize( const jefc_value_t* histogram, 
                     size_t binCount, 
                     /* out */ jefc_value_t* normalizedHistogram );

/*
 * Initializes a vector of values to zero, to start building a arithmetic mean.
 */
void jefc_arithmetic_mean_init( jefc_value_t* mean, size_t vectorSize );

/*
 * Initializes a vector of values to one, to start building a geometric mean.
 */
void jefc_geoemtric_mean_init( jefc_value_t* mean, size_t vectorSize );

/*
 * Given a weight and a vector of values, add it to the arithmetic mean we are building.
 *
 * For a uniform arithmetic mean, weight should be 1 / N, where N is the number of vectors we are calculating a mean for.
 */
void jefc_arithmetic_weighted_mean_step( const jefc_value_t* value, jefc_value_t weight, size_t vectorSize, /*in out*/ jefc_value_t* mean );

/*
 * Given a weight and a vector of values, product it to with the geometric mean we are building.
 *
 * For a uniform geometric mean, weight should be 1 / N, where N is the number of vectors we are calculating a mean for.
 */
void jefc_geometric_weighted_mean_step( const jefc_value_t* value, jefc_value_t weight, size_t vectorSize, /*in out*/ jefc_value_t* mean );


#ifdef __cplusplus
}
#endif


#ifdef JEFC_IMPLEMENTATION

#include <math.h>
#include <limits.h>
#include <float.h>

#define JEFC_MAX_CENTROID_ITERATIONS 100

#ifdef JEFC_DOUBLE 
#define JEFC_LN( X )            log( X )
#define JEFC_EXP( X )           exp( X )
#define JEFC_PRECISION          DBL_MIN
#define JEFC_E                  2.71828182845904523536
#define JEFC_FRITSCH_ITERATIONS 2
#define JEFC_POW( X, Y )        pow( X, Y )
#else
#define JEFC_LN( X )            logf( X )
#define JEFC_EXP( X )           expf( X )
#define JEFC_PRECISION          FLT_MIN
#define JEFC_E                  2.71828182845904523536f
#define JEFC_FRITSCH_ITERATIONS 1
#define JEFC_POW( X, Y )        powf( X, Y )
#endif

/*
 * Random numbers implementation: 
 * Implementation of PCG from the paper PCG: A family of simple fast space-efficient 
 * statistically good algorithms for random number generation http://www.pcg-random.org/pdf/hmc-cs-2014-0905.pdf
 */
#define JEFC_RAND_FACTOR 6364136223846793005ULL
#define JEFC_ROTATEU32( value, rotation ) ( ( ( value ) >> rotation ) | ( ( value ) << ( 32 - rotation ) ) ) 


JEFC_IMPLEMENTATION_TYPE 
uint32_t jefc_random( jefc_random_t* context )
{
    uint64_t previousState = context->state;
    uint32_t mungedState;

    context->state = JEFC_RAND_FACTOR * previousState + context->offset;

    mungedState = (uint32_t)( ( previousState ^ ( previousState >> 18 ) ) >> 27 );

    return JEFC_ROTATEU32( mungedState, (uint32_t)( previousState >> 59 ) );
}


JEFC_IMPLEMENTATION_TYPE 
void jefc_seed( jefc_random_t* context, uint64_t stateSeed, uint64_t offsetSeed )
{
    context->offset = 2 * offsetSeed + 1;
    context->state  = JEFC_RAND_FACTOR * ( stateSeed + context->offset ) + context->offset; 
}


/*
 * Divergence implementations.
 */

JEFC_IMPLEMENTATION_TYPE 
jefc_value_t jefc_jeffreys_divergence( const jefc_value_t* p, const jefc_value_t* q, size_t binCount )
{
    jefc_value_t result = 0;
    size_t       i;

    for ( i = 0; i < binCount; ++i )
    {
        jefc_value_t pi = p[ i ];
        jefc_value_t qi = q[ i ];

        result += ( pi - qi ) * JEFC_LN( pi / qi );
    }

    return result;
}


JEFC_IMPLEMENTATION_TYPE 
jefc_value_t jefc_extended_kl_divergence( const jefc_value_t* p, const jefc_value_t* q, size_t binCount )
{
    jefc_value_t result = 0;
    size_t       i;

    for ( i = 0; i < binCount; ++i )
    {
        jefc_value_t pi = p[ i ];
        jefc_value_t qi = q[ i ];

        result += pi * JEFC_LN( pi / qi ) + ( qi - pi );
    }

    return result;
}


JEFC_IMPLEMENTATION_TYPE 
jefc_value_t jefc_kl_divergence( const jefc_value_t* p, const jefc_value_t* q, size_t binCount )
{
    jefc_value_t result = 0;
    size_t       i;

    for ( i = 0; i < binCount; ++i )
    {
        jefc_value_t pi = p[ i ];
        jefc_value_t qi = q[ i ];

        result += pi * JEFC_LN( pi / qi );
    }

    return result;
}


/*
 * Special case branch 0 lambert W implementation.
 */

 /*
 * Special case z >= 0 lambert W branch zero.
 *
 * Uses the approximation from "Analytical approximations for real values of the Lambert W-function" by Barry et al 
 * to initialize Fritsch's numerical method.
 * 
 */
JEFC_IMPLEMENTATION_TYPE
jefc_value_t jefc_lambert_w0_positive( jefc_value_t z )
{
    double zd               = z; /* just put z at double precision */

    if ( zd == 0.0 ) return (jefc_value_t)0.0;

    double w0;

    if ( zd > JEFC_E )
    {
        double a = log( zd );
        double b = log( a );

        double ia = 1.0 / a;

        double ia2 = ia * ia;
        double ia3 = ia * ia2;
        double ia4 = ia2 * ia2;
        double ia5 = ia4 * ia;

        double b2 = b * b;
        double b3 = b2 * b;
        double b4 = b2 * b2;

        w0 = a - b + 
            b * ( ia + 
                  0.5 * ( -2.0 + b ) * ia2 + 
                  ( 1.0 / 6.0 ) * (  6.0 - 9.0 * b + 2.0 * b2 ) * ia3 + 
                  ( 1.0 / 12.0 ) * ( -12.0 + 36.0 * b - 22.0 * b2 + 3.0 * b3 ) * ia4 +
                  ( 1.0 / 60.0 ) * ( 60.0 - 300.0 * b + 350.0 * b2 - 125.0 * b3 + 12.0 * b4 ) * ia5 );
    }
    else
    {
        double log1zd = log( 1.0 + zd );

        if ( log1zd == 0.0 )
        {
            log1zd = zd;
        }

        double zd2 = zd * zd;

        w0 = log1zd * ( 1.0 + ( 123.0 / 40.0 ) * zd + ( 21.0 / 10.0 ) * zd2 ) / 
            ( 1.0 + ( 143.0 / 40.0 ) * zd + ( 713.0 / 240.0 ) * zd2 );

        if ( w0 < 1e-17 )
        {
            return w0;
        }
    }

    size_t where;

    /* Use 1 iteration of Fristch for float, 2 for double */
    for ( where = 0; where < JEFC_FRITSCH_ITERATIONS; ++where )
    {
        double zn      = log( zd / w0 ) - w0;
        double w0Plus1 = 1.0 + w0;
        double qn      = 2 * ( w0Plus1 ) * ( w0Plus1 + ( 2.0 / 3.0 ) * zn );
        double delta   = w0 * ( zn / w0Plus1 ) * ( ( qn - zn ) / ( qn - 2.0 * zn ) );

        w0 += delta;
    }

    /* Round back to our implementation floating point type (if necessary). */
    return (jefc_value_t)( w0 );
}


/*
 * Jeffreys centroid implementations.
 */

JEFC_IMPLEMENTATION_TYPE 
void jefc_jeffreys_positive_centroid( const jefc_value_t* arithmeticWeightedMean,
                                      const jefc_value_t* geometricWeightedMean,
                                      size_t binCount,
                                      jefc_value_t* centroid )
{
    size_t i;

    for ( i = 0; i < binCount; ++i )
    {
        jefc_value_t ai = arithmeticWeightedMean[ i ];
        jefc_value_t gi = geometricWeightedMean[ i ];

        /* ai and gi should be positive and non-zero */
        centroid[ i ] = ai / jefc_lambert_w0_positive( JEFC_E * ai / gi );
    }
}


JEFC_IMPLEMENTATION_TYPE 
void jefc_jeffreys_frequency_centroid( const jefc_value_t* normalizedArithmeticWeightedMean,
                                       const jefc_value_t* normalizedGeometricWeightedMean,
                                       size_t binCount,
                                       /* out */ jefc_value_t* centroid )
{
    size_t i;
    size_t iteration;

    for ( i = 0; i < binCount; ++i )
    {
        centroid[ i ] = normalizedGeometricWeightedMean[ i ];
    }

    for ( iteration = 0; iteration < JEFC_MAX_CENTROID_ITERATIONS; ++iteration )
    {
        jefc_value_t lambda     = -jefc_extended_kl_divergence( centroid, normalizedGeometricWeightedMean, binCount );
        jefc_value_t exp1lambda = JEFC_EXP( (jefc_value_t)1.0 + lambda );
        size_t       sameCount  = 0;

        for ( i = 0; i < binCount; ++i )
        {
            jefc_value_t ai           = normalizedArithmeticWeightedMean[ i ];
            jefc_value_t gi           = normalizedGeometricWeightedMean[ i ];
            jefc_value_t centroidi    = centroid[ i ];
            jefc_value_t newCentroidi = ai / jefc_lambert_w0_positive( exp1lambda * ai / gi );
            
            sameCount += fabs( centroidi - newCentroidi ) <= JEFC_PRECISION;

            centroid[ i ] = newCentroidi;
        }

        if ( sameCount == binCount ) break;
    }
}


/*
 * Histogram manipulation functions.
 */

JEFC_IMPLEMENTATION_TYPE 
void jefc_smooth( const jefc_value_t* normalizedHistogram,
                  size_t binCount, 
                  float smoothingValue,
                  /* out */ jefc_value_t* smoothedHistogram )
{
    jefc_value_t normalizationValue = (jefc_value_t)1.0 / ( (jefc_value_t)1.0 + ( smoothingValue * binCount ) );
    size_t       where;

    for ( where = 0; where < binCount; ++where )
    {
        smoothedHistogram[ where ] = ( normalizedHistogram[ where ] + smoothingValue ) * normalizationValue;
    }
}


JEFC_IMPLEMENTATION_TYPE
void jefc_normalize( const jefc_value_t* histogram,
                     size_t binCount,
                     /* out */ jefc_value_t* normalizedHistogram )
{
    jefc_value_t total        = 0;
    jefc_value_t inverseTotal;
    size_t       where;

    for ( where = 0; where < binCount; ++where )
    {
        total += histogram[ where ];
    }

    inverseTotal = (jefc_value_t)1.0 / total;

    for ( where = 0; where < binCount; ++where )
    {
        normalizedHistogram[ where ] = histogram[ where ] * inverseTotal;
    }
}


JEFC_IMPLEMENTATION_TYPE
void jefc_normalize_inplace( /* in out */ jefc_value_t* histogram,
                             size_t binCount )
{
    jefc_value_t total        = 0;
    jefc_value_t inverseTotal;
    size_t       where;

    for ( where = 0; where < binCount; ++where )
    {
        total += histogram[ where ];
    }

    inverseTotal = (jefc_value_t)1.0 / total;

    for ( where = 0; where < binCount; ++where )
    {
        histogram[ where ] = histogram[ where ] * inverseTotal;
    }
}


/*
 * Means implementations.
 */

JEFC_IMPLEMENTATION_TYPE
void jefc_arithmetic_mean_init( jefc_value_t* mean, size_t vectorSize )
{
    size_t where;

    for ( where = 0; where < vectorSize; ++where )
    {
        mean[ where ] = 0;
    }
}


JEFC_IMPLEMENTATION_TYPE
void jefc_geoemtric_mean_init( jefc_value_t* mean, size_t vectorSize )
{
    size_t where;

    for ( where = 0; where < vectorSize; ++where )
    {
        mean[ where ] = 1;
    }
}


JEFC_IMPLEMENTATION_TYPE
void jefc_arithmetic_weighted_mean_step( const jefc_value_t* value, jefc_value_t weight, size_t vectorSize, /*in out*/ jefc_value_t* mean )
{
    size_t where;

    for ( where = 0; where < vectorSize; ++where )
    {
        mean[ where ] += value[ where ] * weight;
    }
}


JEFC_IMPLEMENTATION_TYPE
void jefc_geometric_weighted_mean_step( const jefc_value_t* value, jefc_value_t weight, size_t vectorSize, /*in out*/ jefc_value_t* mean )
{
    size_t where;

    for ( where = 0; where < vectorSize; ++where )
    {
        mean[ where ] *= JEFC_POW( value[ where ], weight );
    }
}


JEFC_IMPLEMENTATION_TYPE
jefc_error_t jefc_jeffreys_weighted_kmeans(
    const jefc_kmeans_value_t* values,
    size_t valueCount,
    size_t histogramCount,
    const size_t* binCounts,
    size_t maxIterations,
    uint64_t streamNumber,
    size_t centroidCount,
    /*out*/ jefc_kmeans_centroid_t* centroids,
    /*out*/ jefc_centroid_reference_t* centroidReferences )
{
    if ( centroidCount == 0 || centroidCount > valueCount )
    {
        return JEFC_INVALID_CENTROID_COUNT;
    }

    if ( values == 0 || binCounts == 0 || centroids == 0 || centroidReferences == 0 )
    {
        return JEFC_NULL_PARAMETER;
    }

    jefc_random_t random;

    jefc_seed( &random, 0x575e6be8a29f88e5, streamNumber );

    uint64_t lower       = jefc_random( &random );
    uint64_t upper       = jefc_random( &random );
         
    /* Initial chosen centroid is random.
     * Note, this is a biased random sample due to 0xFFFFFFFF not being wholy divisible by valueCount - but for our purposes, that's good enough. */
    size_t chosenCentroidIndex = (size_t)( ( ( upper << 32 ) | lower ) % (uint64_t)valueCount );
    size_t initedCentroids     = 0;

    /* loop breaks in middle */
    for ( ;; )
    {
        double                  totalDistanceSquared = 0;
        jefc_kmeans_centroid_t* lastCentroid         = centroids + initedCentroids;

        /* Update the centroid references so that the last centroid points to itself as the closest reference.
         * Note, setting the distance to 0 gets around any numerical precision problems for evaluating the distance.
         */
        centroidReferences[ chosenCentroidIndex ].index    = initedCentroids;
        centroidReferences[ chosenCentroidIndex ].distance = 0;
        
        /* Copy the last centroid to the centroids list */
        for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
        {
            lastCentroid->histograms[ histogramIndex ].weight = values[ chosenCentroidIndex ].histograms[ histogramIndex ].weight;

            for ( size_t binIndex = 0; binIndex < binCounts[ histogramIndex ]; ++binIndex )
            {
                lastCentroid->histograms[ histogramIndex ].bins[ binIndex ] = 
                    values[ chosenCentroidIndex ].histograms[ histogramIndex ].bins[ binIndex ];
            }
        }
        
        /* Update the distances to centroids with the newest inited centroid */
        for ( size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex )
        {
            jefc_centroid_reference_t* centroidReference = centroidReferences + valueIndex;

            if ( centroidReference->distance > 0.0 )
            {
                jefc_value_t               distanceToCentroid = 0;
                const jefc_kmeans_value_t* currentValue       = values + valueIndex;

                for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
                {
                    const jefc_kmeans_histogram_t* currentHistogram = currentValue->histograms + histogramIndex;

                    distanceToCentroid +=
                        jefc_jeffreys_divergence( lastCentroid->histograms[ histogramIndex ].bins,
                                                  currentHistogram->bins,
                                                  binCounts[ histogramIndex ] ) * currentHistogram->weight; /* should we be using an exponential weight? */
                }

                if ( lastCentroid == centroids || distanceToCentroid < centroidReference->distance )
                {
                    centroidReference->index    = initedCentroids;
                    centroidReference->distance = distanceToCentroid;
                }
            }
            
            totalDistanceSquared += centroidReference->distance * centroidReference->distance;
        }

        /* We have inited a centroid, so bump the counter */
        ++initedCentroids;

        /* loop termination condition */
        if ( initedCentroids >= centroidCount ) break;

        /* Below here we follow k-means++ initialization procedure:
         *  use distance (Jeffreys divergence) squared to nearest current centroid as probability for choosing 
         *  it as a new centroid. 
         */

        /* We use a double here, because float precision is  too low and will skip values. */
        double randomPicker                = totalDistanceSquared * ( ( (double)jefc_random( &random ) ) / ( (double)0xFFFFFFFF ) );
        double runningTotalDistanceSquared = 0;
        bool   centroidChosen              = false;

        for ( size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex )
        {
            jefc_centroid_reference_t* centroidReference = centroidReferences + valueIndex;

            if ( centroidReference->distance > 0 )
            {
                jefc_value_t distance = centroidReference->distance;

                runningTotalDistanceSquared += distance * distance;

                if ( runningTotalDistanceSquared >= randomPicker )
                {
                    chosenCentroidIndex = valueIndex;
                    centroidChosen      = true;
 
                    break;
                }
            }
        }

        if ( !centroidChosen )
        {
            chosenCentroidIndex = valueCount - 1;
        }
    }

    size_t kmeansIteration = 0;

    for ( ;; )
    {
        /* set counts to zero for each centroid*/
        for ( size_t centroidIndex = 0; centroidIndex < centroidCount; ++centroidIndex )
        {
            jefc_kmeans_centroid_t* centroid = centroids + centroidIndex;

            centroid->valueCount = 0;

            for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
            {
                centroid->histograms[ histogramIndex ].weight = 0;

                jefc_arithmetic_mean_init( centroid->normalizedArithmeticMeans[ histogramIndex ], binCounts[ histogramIndex ] );
                jefc_geoemtric_mean_init( centroid->normalizedGeometricMeans[ histogramIndex ], binCounts[ histogramIndex ] );
            }
        }

        /* accumulate counts for each centroid from the values closest */
        for ( size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex )
        {
            jefc_centroid_reference_t* centroidReference = centroidReferences + valueIndex;
            jefc_kmeans_centroid_t*    centroid          = centroids + centroidReference->index;
            const jefc_kmeans_value_t* value             = values + valueIndex;

            ++centroid->valueCount;

            for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
            {
                centroid->histograms[ histogramIndex ].weight += value->histograms[ histogramIndex ].weight;
            }
        }

        for ( size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex )
        {
            jefc_centroid_reference_t* centroidReference = centroidReferences + valueIndex;
            const jefc_kmeans_value_t* value             = values + valueIndex;
            jefc_kmeans_centroid_t*    centroid          = centroids + centroidReference->index;

            for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
            {
                jefc_value_t histogramWeight = value->histograms[ histogramIndex ].weight / centroid->histograms[ histogramIndex ].weight;

                jefc_arithmetic_weighted_mean_step( 
                    value->histograms[ histogramIndex ].bins, 
                    histogramWeight, 
                    binCounts[ histogramIndex ], 
                    centroid->normalizedArithmeticMeans[ histogramIndex ] );

                jefc_geometric_weighted_mean_step( 
                    value->histograms[ histogramIndex ].bins, 
                    histogramWeight, 
                    binCounts[ histogramIndex ], 
                    centroid->normalizedGeometricMeans[ histogramIndex ] );
            }
        }
 
        /* Work out the new arithmetic and geometric means - note, we do this ahead of iteration termination to make sure they're populated. */
        for ( size_t centroidIndex = 0; centroidIndex < centroidCount; ++centroidIndex )
        {
            jefc_kmeans_centroid_t* centroid = centroids + centroidIndex;

            for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
            {
                jefc_normalize_inplace( centroid->normalizedArithmeticMeans[ histogramIndex ], binCounts[ histogramIndex ] );
                jefc_normalize_inplace( centroid->normalizedGeometricMeans[ histogramIndex ], binCounts[ histogramIndex ] );
            }
        }

        if ( kmeansIteration >= maxIterations ) break;

        ++kmeansIteration;

        /* Calculate the new jeffreys centroids for each item */
        for ( size_t centroidIndex = 0; centroidIndex < centroidCount; ++centroidIndex )
        {
            jefc_kmeans_centroid_t* centroid = centroids + centroidIndex;

            for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
            {
                jefc_jeffreys_frequency_centroid(
                    centroid->normalizedArithmeticMeans[ histogramIndex ],
                    centroid->normalizedGeometricMeans[ histogramIndex ],
                    binCounts[ histogramIndex ],
                    centroid->histograms[ histogramIndex ].bins );
            }
        }

        bool allSame = true;

        /* Calculate the new distances to the centroids/centroid references */
        for ( size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex )
        {
            jefc_centroid_reference_t* centroidReference    = centroidReferences + valueIndex;
            const jefc_kmeans_value_t* currentValue         = values + valueIndex;
            size_t                     oldCentroidReference = centroidReference->index;
            size_t                     closestIndex         = 0;
            jefc_value_t               closestDistance      = 0;

            for ( size_t centroidIndex = 0; centroidIndex < centroidCount; ++centroidIndex )
            {
                jefc_kmeans_centroid_t* centroid           = centroids + centroidIndex;
                jefc_value_t            distanceToCentroid = 0;

                for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
                {
                    const jefc_kmeans_histogram_t* currentHistogram = currentValue->histograms + histogramIndex;

                    distanceToCentroid +=
                        jefc_jeffreys_divergence( centroid->histograms[ histogramIndex ].bins,
                                                  currentHistogram->bins,
                                                  binCounts[ histogramIndex ] ) * currentHistogram->weight; /* should we be using an exponential weight? */
                }

                if ( centroid == centroids || distanceToCentroid < closestDistance )
                {
                    closestIndex    = centroidIndex;
                    closestDistance = distanceToCentroid;
                }
            }

            centroidReference->index    = closestIndex;
            centroidReference->distance = closestDistance;

            allSame &= oldCentroidReference == closestIndex;
        }

        /* Terminate due to stablization */
        if ( allSame && kmeansIteration > 1 )
        {
            break;
        }
    }

    return JEFC_SUCCESS;
}


jefc_error_t jefc_allocate_kmeans_values(
    void* ( *memoryAlloc )( size_t size ),
    size_t valueCount,
    size_t histogramCount,
    const size_t* binCounts,
    /* out */ jefc_kmeans_value_t** values,
    /* out */ jefc_centroid_reference_t** centroidReferences )
{
    size_t totalBins = 0;

    for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
    {
        totalBins += binCounts[ totalBins ];
    }

    size_t allocSize = 
        valueCount * ( sizeof( jefc_kmeans_value_t ) + 
                       sizeof( jefc_centroid_reference_t ) + 
                       histogramCount * sizeof( jefc_kmeans_histogram_t ) + 
                       ( totalBins * sizeof( jefc_value_t ) ) );

    void* memory = memoryAlloc( allocSize );

    if ( memory == 0 )
    {
        return JEFC_ALLOC_FAILED;
    }

    *values             = (jefc_kmeans_value_t*)memory;
    *centroidReferences = (jefc_centroid_reference_t*)( *values + valueCount );

    jefc_kmeans_histogram_t* histograms = (jefc_kmeans_histogram_t*)( *centroidReferences + valueCount );
    jefc_value_t*            binValues  = (jefc_value_t*)( histograms + histogramCount * valueCount );

    for ( size_t valueIndex = 0; valueIndex < valueCount; ++valueCount )
    {
        ( *values )[ valueIndex ].histograms = histograms;

        for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramCount )
        {
            histograms[ histogramIndex ].bins   = binValues;
            histograms[ histogramIndex ].weight = 0;

            binValues += binCounts[ histogramIndex ];
        }

        histograms += histogramCount;
    }

    return JEFC_SUCCESS;
}


jefc_error_t jefc_allocate_kmeans_centroids(
    void* ( *memoryAlloc )( size_t size ),
    size_t centroidCount,
    size_t histogramCount,
    const size_t* binCounts,
    /* out */ jefc_kmeans_centroid_t** centroids )
{
    size_t totalBins = 0;

    for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramIndex )
    {
        totalBins += binCounts[ totalBins ];
    }

    size_t allocSize = 
        centroidCount * ( sizeof( jefc_kmeans_centroid_t ) + 
                          histogramCount * ( sizeof( jefc_kmeans_histogram_t ) + sizeof( jefc_value_t* ) ) +
                          ( 3 * totalBins * sizeof( jefc_value_t ) ) );

    void* memory = memoryAlloc( allocSize );

    if ( memory == 0 )
    {
        return JEFC_ALLOC_FAILED;
    }

    *centroids = (jefc_kmeans_centroid_t*)memory;

    jefc_kmeans_histogram_t* histograms = (jefc_kmeans_histogram_t*)( *centroids + centroidCount );
    jefc_value_t**           averages   = (jefc_value_t**)( histograms + centroidCount * histogramCount );
    jefc_value_t*            binValues  = (jefc_value_t*)( averages + centroidCount * histogramCount * 2 );

    for ( size_t centroidIndex = 0; centroidIndex < centroidCount; ++centroidIndex )
    {
        jefc_kmeans_centroid_t* centroid = ( ( *centroids ) + centroidIndex );

        centroid->histograms                = histograms;
        centroid->normalizedArithmeticMeans = averages;

        averages += histogramCount;

        centroid->normalizedGeometricMeans = averages;

        averages += histogramCount;

        for ( size_t histogramIndex = 0; histogramIndex < histogramCount; ++histogramCount )
        {
            histograms[ histogramIndex ].bins   = binValues;
            histograms[ histogramIndex ].weight = 0;

            size_t binCount = binCounts[ histogramIndex ];

            binValues += binCount;

            centroid->normalizedArithmeticMeans[ histogramIndex ] = binValues;

            binValues += binCount;

            centroid->normalizedGeometricMeans[ histogramIndex ] = binValues;

            binValues += binCount;
        }

        histograms += histogramCount;
    }

    return JEFC_SUCCESS;
}

#endif

#endif // --JEFFREYS_CENTROID_H__