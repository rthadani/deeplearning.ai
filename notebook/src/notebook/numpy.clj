(ns notebook.numpy
  (:use [uncomplicate.neanderthal core native vect-math]))

(defn broadcast-num
  [mtx num]
  (let [rows (mrows mtx)
        cols (ncols mtx)]
    (dge rows cols (repeat (* rows cols) num) {:layout :row})))

(defn broadcast-cols
  [mtx bcast-mtx]
  (let [rows (mrows mtx)
        cols (ncols mtx)
        y (zero mtx)]
    (doseq [i (range rows)
            j (range cols)]
      (entry! y i j (entry bcast-mtx i 0)))
    y))

(defn broadcast-rows
  [mtx bcast-mtx]
  (let [rows (mrows mtx)
        cols (ncols mtx)
        y (zero mtx)]
    (doseq [i (range rows)
            j (range cols)]
      (entry! y i j (entry bcast-mtx 0 j)))
    y))

(defn broadcast-mtx
  [mtx bcast-mtx]
  (println mtx bcast-mtx)
  (cond
    (and (= 1 (ncols bcast-mtx)) (= (mrows mtx) (mrows bcast-mtx)))
    (broadcast-cols mtx bcast-mtx)
    (and (= 1 (mrows bcast-mtx)) (= (ncols mtx) (ncols bcast-mtx)))
    (broadcast-rows mtx bcast-mtx)
    :else
    (throw (Exception. "Rows and columns are incompatible"))))

(defn broadcast
  [mtx bcast]
  (if (number? bcast)
    (broadcast-num mtx bcast)
    (broadcast-mtx mtx bcast)))

(defn bcast-op
  [mtx v op]
  (if (and (matrix? v)
           (= (ncols mtx) (ncols v))
           (= (mrows mtx) (mrows v)))
    (op mtx v)
    (op mtx (broadcast mtx v))))

(defn bcast-plus
  [mtx v]
  (bcast-op mtx v xpy))

(defn bcast-mul
  [mtx v]
  (bcast-op mtx v mul))

(defn sum-axis
  [mtx axis]
  (let [row-col-fn (case axis
                     :row rows
                     :col cols)
        sum (map sum (row-col-fn mtx))]
    (case axis
      :col (dge 1 (ncols mtx) sum)
      :row (dge (mrows mtx) 1 sum))) )

#_(def A (dge 3 4 [56.0 0.0 4.4 68.0
                   1.2 104.0 52.0 8.0
                   1.8 135.0 99.0 0.9] ))
#_(broadcast A 1)
#_(def bcast (dge 1 4 [1 2 3 4] ))
#_(def bcast-col (dge 3 1 [1 2 3] ))
#_(broadcast A bcast)
#_(bcast-plus A bcast)
#_(bcast-plus A bcast-col)
#_ (bcast-mul A 1)




