(defproject notebook "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :main ^:skip-aot gorilla-test.core
  :target-path  "target/%s"
  :plugins      [[k2n/lein-gorilla "0.4.1" :exclusions [[org.clojure/clojure]]]]
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [net.mikera/imagez "0.12.0"]
                 [uncomplicate/neanderthal "0.17.2"]]
  :resources ["resources"])
