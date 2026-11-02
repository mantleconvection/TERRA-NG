

#pragma once

#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace terra::util {

class XML
{
  public:
    explicit XML(
        std::string                                 name,
        const std::map< std::string, std::string >& attributes = {},
        std::string                                 content    = "" )
    : name_( std::move( name ) )
    , content_( std::move( content ) )
    {
        attributes_.insert( attributes.begin(), attributes.end() );
    }

    XML& add_child( const XML& child )
    {
        children_.push_back( child );
        return *this;
    }

    [[nodiscard]] std::string to_string() const { return to_string( 0 ); }

    /// @brief Inverse of the escaping done in to_string()/escape_xml().
    [[nodiscard]] static std::string unescape_xml( const std::string& data )
    {
        std::string result;
        result.reserve( data.size() );
        for ( size_t i = 0; i < data.size(); )
        {
            if ( data.compare( i, 5, "&amp;" ) == 0 )
            {
                result += '&';
                i += 5;
            }
            else if ( data.compare( i, 6, "&quot;" ) == 0 )
            {
                result += '"';
                i += 6;
            }
            else if ( data.compare( i, 6, "&apos;" ) == 0 )
            {
                result += '\'';
                i += 6;
            }
            else if ( data.compare( i, 4, "&lt;" ) == 0 )
            {
                result += '<';
                i += 4;
            }
            else if ( data.compare( i, 4, "&gt;" ) == 0 )
            {
                result += '>';
                i += 4;
            }
            else
            {
                result += data[i];
                i += 1;
            }
        }
        return result;
    }

  private:
    std::string                          name_;
    std::string                          content_;
    std::vector< XML >                   children_;
    std::map< std::string, std::string > attributes_;

    [[nodiscard]] std::string to_string( int indent ) const
    {
        std::ostringstream oss;
        std::string        indent_str( indent, ' ' );

        oss << indent_str << "<" << name_;
        for ( const auto& attr : attributes_ )
        {
            oss << " " << attr.first << "=\"" << escape_xml( attr.second ) << "\"";
        }

        if ( children_.empty() && content_.empty() )
        {
            oss << " />\n";
        }
        else
        {
            oss << ">";
            if ( !content_.empty() )
            {
                oss << escape_xml( content_ );
            }
            if ( !children_.empty() )
            {
                oss << "\n";
                for ( const auto& child : children_ )
                {
                    oss << child.to_string( indent + 2 );
                }
                oss << indent_str;
            }
            oss << "</" << name_ << ">\n";
        }

        return oss.str();
    }

    static std::string escape_xml( const std::string& data )
    {
        std::ostringstream escaped;
        for ( char c : data )
        {
            switch ( c )
            {
            case '&':
                escaped << "&amp;";
                break;
            case '\"':
                escaped << "&quot;";
                break;
            case '\'':
                escaped << "&apos;";
                break;
            case '<':
                escaped << "&lt;";
                break;
            case '>':
                escaped << "&gt;";
                break;
            default:
                escaped << c;
                break;
            }
        }
        return escaped.str();
    }
};

/// @brief Finds a single attribute's value in raw, already-serialized XML.
///
/// Not a general-purpose parser - intended only for reading back small pieces of data
/// written by \ref XML::to_string(). Searches for the first occurrence of
/// `<tag_name> ... attribute_name="...">`.
///
/// @return the unescaped attribute value, or std::nullopt if the tag or attribute was not found.
[[nodiscard]] inline std::optional< std::string >
    find_xml_attribute( const std::string& content, const std::string& tag_name, const std::string& attribute_name )
{
    const auto tag_pos = content.find( "<" + tag_name );
    if ( tag_pos == std::string::npos )
        return std::nullopt;

    const auto tag_end = content.find( ">", tag_pos );
    if ( tag_end == std::string::npos )
        return std::nullopt;

    const auto key           = attribute_name + "=\"";
    const auto value_key_pos = content.find( key, tag_pos );
    if ( value_key_pos == std::string::npos || value_key_pos > tag_end )
        return std::nullopt;

    const auto value_start = value_key_pos + key.size();
    const auto value_end   = content.find( '"', value_start );
    if ( value_end == std::string::npos )
        return std::nullopt;

    return XML::unescape_xml( content.substr( value_start, value_end - value_start ) );
}

} // namespace terra::util
