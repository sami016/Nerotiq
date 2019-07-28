namespace Nerotiq.Util {
    public static class MatrixHelpers {
        /**
         * Gets the cardinality of a dimensionality array.
         **/
        public static int GetCardinality(ushort[] dimensionality) {
            var acc = 1;
            foreach(var i in dimensionality) 
            {
                acc *= i;
            }
            return acc;
        }
        
        /**
         * Gets the cardinality of a weight matrix given two dimensionality arrays.
         **/
        public static int GetWeightCardinality(ushort[] fromDimensionality, ushort[] toDimensionality) {
            return GetCardinality(fromDimensionality) 
                * GetCardinality(toDimensionality);
        }
    }
}