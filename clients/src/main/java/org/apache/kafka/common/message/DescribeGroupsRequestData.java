/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// THIS CODE IS AUTOMATICALLY GENERATED.  DO NOT EDIT.

package org.apache.kafka.common.message;

import java.util.ArrayList;
import java.util.List;
import org.apache.kafka.common.errors.UnsupportedVersionException;
import org.apache.kafka.common.protocol.ApiMessage;
import org.apache.kafka.common.protocol.MessageUtil;
import org.apache.kafka.common.protocol.Readable;
import org.apache.kafka.common.protocol.Writable;
import org.apache.kafka.common.protocol.types.ArrayOf;
import org.apache.kafka.common.protocol.types.Field;
import org.apache.kafka.common.protocol.types.Schema;
import org.apache.kafka.common.protocol.types.Struct;
import org.apache.kafka.common.protocol.types.Type;


public class DescribeGroupsRequestData implements ApiMessage {
    private List<String> groups;
    private boolean includeAuthorizedOperations;
    
    public static final Schema SCHEMA_0 =
        new Schema(
            new Field("groups", new ArrayOf(Type.STRING), "The names of the groups to describe")
        );
    
    public static final Schema SCHEMA_1 = SCHEMA_0;
    
    public static final Schema SCHEMA_2 = SCHEMA_1;
    
    public static final Schema SCHEMA_3 =
        new Schema(
            new Field("groups", new ArrayOf(Type.STRING), "The names of the groups to describe"),
            new Field("include_authorized_operations", Type.BOOLEAN, "Whether to include authorized operations.")
        );
    
    public static final Schema[] SCHEMAS = new Schema[] {
        SCHEMA_0,
        SCHEMA_1,
        SCHEMA_2,
        SCHEMA_3
    };
    
    public DescribeGroupsRequestData(Readable readable, short version) {
        this.groups = new ArrayList<String>();
        read(readable, version);
    }
    
    public DescribeGroupsRequestData(Struct struct, short version) {
        this.groups = new ArrayList<String>();
        fromStruct(struct, version);
    }
    
    public DescribeGroupsRequestData() {
        this.groups = new ArrayList<String>();
        this.includeAuthorizedOperations = false;
    }
    
    @Override
    public short apiKey() {
        return 15;
    }
    
    @Override
    public short lowestSupportedVersion() {
        return 0;
    }
    
    @Override
    public short highestSupportedVersion() {
        return 3;
    }
    
    @Override
    public void read(Readable readable, short version) {
        {
            int arrayLength = readable.readInt();
            if (arrayLength < 0) {
                this.groups = null;
            } else {
                this.groups.clear();
                for (int i = 0; i < arrayLength; i++) {
                    this.groups.add(readable.readNullableString());
                }
            }
        }
        if (version >= 3) {
            this.includeAuthorizedOperations = readable.readByte() != 0;
        } else {
            this.includeAuthorizedOperations = false;
        }
    }
    
    @Override
    public void write(Writable writable, short version) {
        writable.writeInt(groups.size());
        for (String element : groups) {
            writable.writeString(element);
        }
        if (version >= 3) {
            writable.writeByte(includeAuthorizedOperations ? (byte) 1 : (byte) 0);
        }
    }
    
    @Override
    public void fromStruct(Struct struct, short version) {
        {
            Object[] nestedObjects = struct.getArray("groups");
            this.groups = new ArrayList<String>(nestedObjects.length);
            for (Object nestedObject : nestedObjects) {
                this.groups.add((String) nestedObject);
            }
        }
        if (version >= 3) {
            this.includeAuthorizedOperations = struct.getBoolean("include_authorized_operations");
        } else {
            this.includeAuthorizedOperations = false;
        }
    }
    
    @Override
    public Struct toStruct(short version) {
        Struct struct = new Struct(SCHEMAS[version]);
        {
            String[] nestedObjects = new String[groups.size()];
            int i = 0;
            for (String element : this.groups) {
                nestedObjects[i++] = element;
            }
            struct.set("groups", (Object[]) nestedObjects);
        }
        if (version >= 3) {
            struct.set("include_authorized_operations", this.includeAuthorizedOperations);
        }
        return struct;
    }
    
    @Override
    public int size(short version) {
        int size = 0;
        size += 4;
        for (String element : groups) {
            size += 2;
            size += MessageUtil.serializedUtf8Length(element);
        }
        if (version >= 3) {
            size += 1;
        } else {
            if (includeAuthorizedOperations) {
                throw new UnsupportedVersionException("Attempted to write a non-default includeAuthorizedOperations at version " + version);
            }
        }
        return size;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof DescribeGroupsRequestData)) return false;
        DescribeGroupsRequestData other = (DescribeGroupsRequestData) obj;
        if (this.groups == null) {
            if (other.groups != null) return false;
        } else {
            if (!this.groups.equals(other.groups)) return false;
        }
        if (includeAuthorizedOperations != other.includeAuthorizedOperations) return false;
        return true;
    }
    
    @Override
    public int hashCode() {
        int hashCode = 0;
        hashCode = 31 * hashCode + (groups == null ? 0 : groups.hashCode());
        hashCode = 31 * hashCode + (includeAuthorizedOperations ? 1231 : 1237);
        return hashCode;
    }
    
    @Override
    public String toString() {
        return "DescribeGroupsRequestData("
            + "groups=" + MessageUtil.deepToString(groups.iterator())
            + ", includeAuthorizedOperations=" + (includeAuthorizedOperations ? "true" : "false")
            + ")";
    }
    
    public List<String> groups() {
        return this.groups;
    }
    
    public boolean includeAuthorizedOperations() {
        return this.includeAuthorizedOperations;
    }
    
    public DescribeGroupsRequestData setGroups(List<String> v) {
        this.groups = v;
        return this;
    }
    
    public DescribeGroupsRequestData setIncludeAuthorizedOperations(boolean v) {
        this.includeAuthorizedOperations = v;
        return this;
    }
}
