#pragma once

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "logging.hpp"
#include "result.hpp"
#include "terra/mpi/mpi.hpp"
#include "timestamp.hpp"

namespace terra::util {

/// @brief Table class for storing and manipulating tabular data.
///
/// Rows are stored as maps from column names to values.
/// Columns are dynamically added as needed.
/// Think: each row is a dict-like type. Columns are essentially keys.
///
/// @note Each row automatically gets an "id" and "timestamp" column.
///
/// Provides functionality to add rows, select columns, query data, and print in various formats (pretty, JSON lines, CSV).
///
/// Not optimized for performance but designed for ease of use and flexibility.
/// If you need high performance, or millions of rows, consider using a database or specialized library.
/// But still useful for small to medium datasets, logging, prototyping, and data analysis tasks.
///
/// Supports various value types, including strings, numbers, booleans, and None (null).
///
/// @note For logging, the convention is that most functions use the key "tag" to add some keyword to the table. To
///       later sort data, therefore add a "tag" to mark where the data comes from.
///
/// Example usage:
/// @code
///
/// terra::util::Table table;
///
/// // Add rows with various data types.
/// table.add_row({{"name", "Charlie"}, {"age", 28}, {"active", true}});
/// table.add_row({{"name", "Alice"}, {"age", 30}, {"active", false}});
/// table.add_row({{"name", "Bob"}, {"age", 25}, {"active", true}});
///
/// // Add rows with different columns to the same table.
/// table.add_row({{"city", "Berlin"}, {"country", "Germany"}, {"population", 3769000}});
///
/// // Select specific columns
/// auto selected_table = table.select_columns({ "name", "age" });
///
/// // Query rows where age is greater than 26
/// auto queried_table = table.query_rows_where( "age", []( const auto& value ) {
///     return std::holds_alternative<int>( value ) && std::get<int>( value ) > 26;
/// });
///
/// // Print the table in different formats
/// table.print_pretty();
/// selected_table.print_jsonl();
/// queried_table.print_csv();
///
/// // Print to file
///
/// // CSV
/// std::ofstream file("output.csv");
/// table.print_csv(file);
/// file.close();
///
/// // JSON lines
/// std::ofstream json_file("output.jsonl");
/// table.print_jsonl(json_file);
/// json_file.close();
///
/// // Clear the table
/// table.clear();
/// @endcode
///
class Table
{
  public:
    /// @brief Max length of string values (required for safe reading of possibly non-null-terminated char arrays).
    static constexpr int MAX_STRING_LENGTH = 10000;

  private:
    using ValueBase = std::variant<
        std::monostate,
        std::string,
        char,
        short,
        int,
        long,
        long long,
        unsigned char,
        unsigned short,
        unsigned int,
        unsigned long,
        unsigned long long,
        float,
        double,
        bool >;

  public:
    /// @brief Type for table cell values.
    struct Value : ValueBase
    {
        // Using a variant directly is annoying because we need to handle string literals via const char *.
        // (That conversion btw seems to be compiler-dependent :) which makes it even more annoying.)
        // It is possible but requires another special case throughout the accessors.
        // So we simply always convert to std::string.

        using ValueBase::ValueBase; // inherit constructors

        Value( const char* arg )
        : ValueBase( char_ptr_to_string_safe( arg ) )
        {}
    };

    /// @brief Type for a table row (mapping column name to value).
    using Row = std::unordered_map< std::string, Value >;

    /// @brief Construct an empty table.
    Table() = default;

    /// @brief Get all rows in the table.
    /// @return Vector of rows.
    [[nodiscard]] const std::vector< Row >& rows() const { return rows_; }

    /// @brief Get all column names in the table.
    /// @return Set of column names.
    [[nodiscard]] const std::set< std::string >& columns() const { return columns_; }

    /// @brief Add a row to the table.
    /// Adds "id" and "timestamp" columns automatically.
    /// @param row_data Row data as a map from column name to value.
    void add_row( const Row& row_data )
    {
        Row row;
        row["id"]        = global_id_counter++;
        row["timestamp"] = current_timestamp();
        columns_.insert( "id" );
        columns_.insert( "timestamp" );

        for ( const auto& [key, value] : row_data )
        {
            row[key] = value;
            columns_.insert( key );
        }

        rows_.emplace_back( row );
    }
    
    /// @brief Set specific column ordering by key.
    /// @param order List of key elements in order of choice.
    void set_column_order( const std::vector< std::string >& order )
    {
        column_order_ = order;
    }

    /// @brief Select a subset of columns from the table.
    /// @param selected_columns Columns to select.
    /// @return New Table with only selected columns.
    [[nodiscard]] Table select_columns( const std::vector< std::string >& selected_columns ) const
    {
        Table result;

        for ( const auto& col : selected_columns )
        {
            result.columns_.insert( col );
        }

        for ( const auto& row : rows_ )
        {
            Row new_row;
            for ( const auto& col : result.columns_ )
            {
                new_row[col] = get_value_from_row_or_none( row, col );
            }
            result.rows_.push_back( std::move( new_row ) );
        }

        return result;
    }

    /// @brief Query rows where a column is not None.
    /// @param column Column name.
    /// @return New Table with matching rows.
    [[nodiscard]] Table query_rows_not_none( const std::string& column ) const
    {
        Table result;
        result.columns_ = columns_;
        for ( const auto& row : rows_ )
        {
            if ( auto it = row.find( column );
                 it != row.end() && !std::holds_alternative< std::monostate >( it->second ) )
            {
                result.rows_.push_back( row );
            }
        }
        return result;
    }

    /// @brief Query rows where a column equals a value.
    /// @param column Column name.
    /// @param value Value to compare.
    /// @return New Table with matching rows.
    [[nodiscard]] Table query_rows_equals( const std::string& column, const Value& value ) const
    {
        Table result;
        result.columns_ = columns_;
        for ( const auto& row : rows_ )
        {
            auto it = row.find( column );
            if ( it != row.end() && it->second == value )
            {
                result.rows_.push_back( row );
            }
        }
        return result;
    }

    /// @brief Query rows where a column satisfies a predicate.
    /// @param column Column name.
    /// @param predicate Predicate function.
    /// @return New Table with matching rows.
    Table query_rows_where( const std::string& column, const std::function< bool( const Value& ) >& predicate ) const
    {
        Table result;
        result.columns_ = columns_;
        for ( const auto& row : rows_ )
        {
            auto it = row.find( column );
            if ( it != row.end() && predicate( it->second ) )
            {
                result.rows_.push_back( row );
            }
        }
        return result;
    }

    /// @brief Returns the values of a column in a vector.
    ///
    /// Casts arithmetic values (std::is_arithmetic_v) if the output type is also arithmetic. Skips the row otherwise.
    /// Skips rows where there is no entry for that column.
    ///
    /// @param column Column name.
    /// @return Vector filled with non-empty values of the column with the specified name.
    template < typename RawType >
    [[nodiscard]] std::vector< RawType > column_as_vector( const std::string& column ) const
    {
        std::vector< RawType > result;

        for ( const auto& row : rows_ )
        {
            auto it = row.find( column );
            if ( it == row.end() )
            {
                continue;
            }

            const auto& val = it->second;

            std::visit(
                [&]( auto&& v ) {
                    using V = std::decay_t< decltype( v ) >;

                    // Case 1 — Exact match
                    if constexpr ( std::is_same_v< V, RawType > )
                    {
                        result.push_back( v );
                    }
                    // Case 2 — allow numeric → numeric
                    else if constexpr ( std::is_arithmetic_v< RawType > && std::is_arithmetic_v< V > )
                    {
                        result.push_back( static_cast< RawType >( v ) );
                    }
                    // Case 3 — string (or anything else) → skip
                    else
                    {
                        // skip
                    }
                },
                val );
        }

        return result;
    }

    /// @brief Print the table in a pretty formatted style.
    /// @param os Output stream (default util::logroot).
    /// @return Reference to this table.
    const Table& print_pretty( std::ostream& os = logroot ) const
    {
        if ( columns_.empty() || rows_.empty() )
        {
            os << "Empty table.\n";
            return *this;
        }

        if ( mpi::rank() == 0 )
        {
            std::unordered_map< std::string, size_t > widths;
            for ( const auto& col : ordered_columns() )
            {
                widths[col] = col.size();
            }

            for ( const auto& row : rows_ )
            {
                for ( const auto& col : ordered_columns() )
                {
                    widths[col] =
                        std::max( widths[col], value_to_string( get_value_from_row_or_none( row, col ) ).size() );
                }
            }

            auto sep = [&] {
                for ( const auto& col : ordered_columns() )
                {
                    os << "+" << std::string( widths[col] + 2, '-' );
                }
                os << "+\n";
            };

            sep();
            os << "|";
            for ( const auto& col : ordered_columns() )
            {
                os << " " << std::setw( static_cast< int >( widths[col] ) ) << std::right << col << " |";
            }
            os << "\n";
            sep();

            for ( const auto& row : rows_ )
            {
                os << "|";
                for ( const auto& col : ordered_columns() )
                {
                    os << " " << std::setw( static_cast< int >( widths[col] ) ) << std::right
                       << value_to_string( get_value_from_row_or_none( row, col ) ) << " |";
                }
                os << "\n";
            }
            sep();
        }

        return *this;
    }

    /// @brief Print the table as JSON lines.
    /// Each row is a JSON object, one per line.
    ///
    /// Example output:
    /// {"id":1,"timestamp":"2024-06-10 12:34:56","name":"Alice","age":30}
    /// {"id":2,"timestamp":"2024-06-10 12:34:57","name":"Bob","age":25}
    ///
    /// To parse with pandas:
    /// import pandas as pd
    /// df = pd.read_json("yourfile.jsonl", lines=True)
    ///
    /// @param os Output stream (default util::logroot).
    /// @return Reference to this table.
    const Table& print_jsonl( std::ostream& os = logroot ) const
    {
        if ( mpi::rank() == 0 )
        {
            for ( const auto& row : rows_ )
            {
                os << "{";
                bool first = true;
                for ( const auto& [key, val] : row )
                {
                    if ( !first )
                    {
                        os << ",";
                    }
                    os << "\"" << key << "\":";
                    if ( std::holds_alternative< std::string >( val ) )
                    {
                        os << "\"" << value_to_string( val ) << "\"";
                    }
                    else if ( std::holds_alternative< std::monostate >( val ) )
                    {
                        os << "null";
                    }
                    else if ( std::holds_alternative< bool >( val ) )
                    {
                        os << ( std::get< bool >( val ) ? "true" : "false" );
                    }
                    else
                    {
                        os << value_to_string( val );
                    }
                    first = false;
                }

                os << "}\n";
            }
        }

        return *this;
    }

    /// @brief Print the table as CSV.
    /// @param os Output stream (default util::logroot).
    /// @return Reference to this table.
    const Table& print_csv( std::ostream& os = logroot ) const
    {
        if ( mpi::rank() == 0 )
        {
            print_header( os, "," );
            for ( const auto& row : rows_ )
            {
                bool first = true;
                for ( const auto& col : ordered_columns() )
                {
                    if ( !first )
                        os << ",";
                    os << value_to_string( get_value_from_row_or_none( row, col ) );
                    first = false;
                }
                os << "\n";
            }
        }

        return *this;
    }

    /// @brief Clear all rows and columns from the table.
    ///
    /// This resets the table to an empty state - but does not reset the running id counter.
    void clear()
    {
        rows_.clear();
        columns_.clear();
    }

    /// @brief Convert a Value to a string for printing.
    /// @param v Value to convert.
    /// @return String representation.
    static std::string value_to_string( const Value& v )
    {
        return std::visit(
            []( const auto& val ) -> std::string {
                using T = std::decay_t< decltype( val ) >;
                if constexpr ( std::is_same_v< T, std::monostate > )
                {
                    return "None";
                }
                else if constexpr ( std::is_same_v< T, std::string > )
                {
                    return val;
                }
                else if constexpr ( std::is_same_v< T, bool > )
                {
                    return val ? "true" : "false";
                }
                else if constexpr ( std::is_same_v< T, float > || std::is_same_v< T, double > )
                {
                    std::ostringstream ss;
                    ss << std::scientific << std::setprecision( 3 ) << val;
                    return ss.str();
                }
                else
                {
                    return std::to_string( val );
                }
            },
            v );
    }

    /// @brief Get a value from a row or return None if missing.
    /// @param row Row to search.
    /// @param col Column name.
    /// @return Value or None.
    static Value get_value_from_row_or_none( const Row& row, const std::string& col )
    {
        auto it = row.find( col );
        return ( it != row.end() ) ? it->second : std::monostate{};
    }

  private:
    std::vector< Row >      rows_;    ///< Rows of the table.
    std::set< std::string > columns_; ///< Set of column names.
    std::optional< std::vector< std::string > > column_order_; ///< Order of column names.

    inline static int global_id_counter = 0; ///< Global row id counter.

    /// @brief Helper function to manage specifically ordered and unspecified (i.e. alphabetically ordered columns.
    const std::vector< std::string > ordered_columns() const
    {
        if ( column_order_.has_value() )
	    return column_order_.value();
	return std::vector< std::string >( columns_.begin(), columns_.end() );
    }

    /// @brief Print the table header.
    /// @param os Output stream.
    /// @param sep Separator string.
    void print_header( std::ostream& os, const std::string& sep ) const
    {
        bool first = true;
        for ( const auto& col : ordered_columns() )
        {
            if ( !first )
            {
                os << sep;
            }

            os << col;
            first = false;
        }
        os << "\n";
    }

    /// @brief Safely converts const char * to string, even if not null terminated (constant max string length of MAX_STRING_LENGTH).
    static std::string char_ptr_to_string_safe( const char* val )
    {
        if ( !val )
        {
            return std::string{};
        }

        std::size_t len = strnlen( val, MAX_STRING_LENGTH ); // find '\0' but stop at MAX_LEN
        return { val, len };
    }
};

/// @brief Attempts to read a csv file and converts that into a \ref Table instance.
///
/// Recognizes commas as a separator.
///
/// @param filename CSV file path
/// @return Result object either containing a \ref Table on success or an error string.
[[nodiscard]] inline Result< Table > read_table_from_csv( const std::string& filename )
{
    auto trim = []( const std::string& s ) {
        size_t start = 0;
        while ( start < s.size() && std::isspace( static_cast< unsigned char >( s[start] ) ) )
        {
            start++;
        }
        size_t end = s.size();
        while ( end > start && std::isspace( static_cast< unsigned char >( s[end - 1] ) ) )
        {
            end--;
        }
        return s.substr( start, end - start );
    };

    auto parse_line = [trim]( const std::string& line ) {
        std::vector< std::string > result;
        std::string                field;
        bool                       inQuotes = false;

        for ( size_t i = 0; i < line.size(); ++i )
        {
            if ( const char c = line[i]; c == '"' )
            {
                if ( inQuotes && i + 1 < line.size() && line[i + 1] == '"' )
                {
                    field.push_back( '"' ); // escaped quote
                    ++i;
                }
                else
                {
                    inQuotes = !inQuotes;
                }
            }
            else if ( c == ',' && !inQuotes )
            {
                result.push_back( trim( field ) );
                field.clear();
            }
            else
            {
                field.push_back( c );
            }
        }

        result.push_back( trim( field ) );
        return result;
    };

    auto infer_value = []( const std::string& s ) -> Table::Value {
        if ( s == "true" || s == "TRUE" )
            return true;
        if ( s == "false" || s == "FALSE" )
            return false;

        char* end = nullptr;

        // try int64
        long long iv = std::strtoll( s.c_str(), &end, 10 );
        if ( end && *end == '\0' )
        {
            return static_cast< int64_t >( iv );
        }

        // try double
        double dv = std::strtod( s.c_str(), &end );
        if ( end && *end == '\0' )
        {
            return dv;
        }

        // fallback to string
        return s;
    };

    auto build_row = [infer_value](
                         const std::vector< std::string >& headers,
                         const std::vector< std::string >& fields ) -> Table::Row {
        Table::Row row;
        for ( size_t i = 0; i < headers.size() && i < fields.size(); ++i )
        {
            row[headers[i]] = infer_value( fields[i] );
        }
        return row;
    };

    std::ifstream file( filename );
    if ( !file.is_open() )
    {
        return { "Could not open file: " + filename };
    }

    std::string line;
    if ( !std::getline( file, line ) )
    {
        return { "Could not find a single line of content in file: " + filename };
    }

    std::vector< std::string > headers = parse_line( line );

    Table table;

    long line_number = 2;
    while ( std::getline( file, line ) )
    {
        auto fields = parse_line( line );

        if ( fields.size() != headers.size() )
        {
            return {
                "Error parsing CSV file (line " + std::to_string( line_number ) +
                "):\n"
                "Number of columns in row does not match number of headers.\n"
                "Note that also empty lines could be the cause of this (an empty line has one column with empty value).\n"
                "Line:" +
                line };
        }

        auto row = build_row( headers, fields );
        table.add_row( row );

        line_number++;
    }

    return table;
}

} // namespace terra::util
