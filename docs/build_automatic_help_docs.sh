#!/bin/bash

# This generates the automatic pages with the --help
# This is added in

index_page=docs/automatic_doc/index_automatic.rst

# Preparing the toctree
if [ -f $index_page ]
then
    rm $index_page
fi
{
  echo "Scripts"
  echo "======="
  echo " "
  echo ".. toctree::"
  echo "    :caption: Scripts (--help)"
  echo " "
} > $index_page


for script in src/dwi_ml/cli/*.py;
do
    script_name=$(basename $script)
    script_name=${script_name%.py}

    if [ $script_name != '__init__' ]
    then
      echo "Preparing help page for script $script_name"

      # Prepare help file
      outfile="docs/automatic_doc/${script_name}_help.rst"
      if [ -f $outfile ]
      then
          rm $outfile
      fi
      {
        echo $script_name
        echo "========================================================="
        echo ""
        echo ".. code-block:: text"
        echo ""
        $script_name --help | sed 's/^/    /'
      } > $outfile

      # Add file to toctree
      echo "    ${script_name}_help" >> $index_page
    fi
done