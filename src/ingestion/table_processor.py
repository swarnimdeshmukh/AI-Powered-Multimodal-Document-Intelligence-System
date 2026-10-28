"""Convert extracted tables into stable searchable representations."""

import pandas as pd


class TableProcessor:
    def to_dataframe(self, table):
        return pd.DataFrame(table)

    def to_markdown(self, table):
        return self.to_dataframe(table).to_markdown(index=False)
